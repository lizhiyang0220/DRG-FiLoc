import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import contextlib
import math
import random
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Activation,
    GlobalAveragePooling2D, Dropout, Conv2D, Multiply,
    GaussianNoise, Concatenate, Add, Rescaling,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


REGIONS = [
    {"name": "CAR",   "train": "wifi_training_CAR.csv",   "test": "wifi_evaluation_CAR.csv"},
    {"name": "UAH",   "train": "wifi_training_UAH.csv",   "test": "wifi_evaluation_UAH.csv"},
    {"name": "UJITI", "train": "wifi_training_UJITI.csv", "test": "wifi_evaluation_UJITI.csv"},
    {"name": "UJIUB", "train": "wifi_training_UJIUB.csv", "test": "wifi_evaluation_UJIUB.csv"},
]
LABEL_COLS = ("x_m", "y_m")
SEED = 42
VAL_RATIO = 0.20
EPOCHS = 800
BATCH_SIZE = 64
RSSI_MISSING = -110.0
RSSI_SCALE_MIN = -110.0
RSSI_SCALE_MAX = 0.0
N_AP = 816
OUTPUT_ROOT = "IPIN2016_DRG-FiLoc"
plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"[WARN] GPU memory growth unavailable: {exc}")
    strategy = (tf.distribute.MirroredStrategy()
                if len(gpus) > 1 else tf.distribute.get_strategy())
    print(f"[INFO] GPUs={len(gpus)}, replicas={strategy.num_replicas_in_sync}")
    return strategy


def sanitize_rssi(values):
    x = np.asarray(values, dtype=np.float32)
    return np.nan_to_num(
        x, nan=RSSI_MISSING, posinf=RSSI_MISSING,
        neginf=RSSI_MISSING,
    ).astype(np.float32)


def rssi_to_unit(values):
    x = sanitize_rssi(values)
    unit = (x - RSSI_SCALE_MIN) / (RSSI_SCALE_MAX - RSSI_SCALE_MIN)
    return np.clip(unit, 0.0, 1.0).astype(np.float32)


def read_and_sort_csv(path, require_file=True):
    df = pd.read_csv(path)
    if len(df) < 1:
        raise ValueError(f"CSV file contains no data rows: {path}")
    if "Timestamp" not in df.columns:
        raise ValueError(f"CSV file is missing the Timestamp column: {path}")
    if "file" not in df.columns:
        if require_file:
            raise ValueError(f"Training CSV is missing the file column: {path}")
        df = df.copy()
        df["file"] = "EVA_SINGLE_LOG"
    df = df.copy()
    df["source_row_index"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(["file", "Timestamp"], kind="mergesort").reset_index(drop=True)
    return df


def load_training_dataset(path, label_cols=LABEL_COLS):
    df = read_and_sort_csv(path, require_file=True)
    if len(df) < 2:
        raise ValueError(f"At least two training samples are required: {path}")
    if any(c not in df.columns for c in label_cols):
        raise ValueError(f"Missing training label columns: {label_cols}")


    if df.shape[1] < 818:
        raise ValueError("The training data must have at least 818 columns to extract 816 AP features.")
    feature_cols = [c for c in list(df.columns)[2:818] if c not in label_cols]
    if len(feature_cols) != N_AP or len(set(feature_cols)) != N_AP:
        raise ValueError(
            f"Expected {N_AP} distinct AP columns among columns 3-818; "
            f"found {len(feature_cols)}. Check the original column layout."
        )
    x = sanitize_rssi(df[feature_cols].to_numpy(dtype=np.float32))
    y = df[list(label_cols)].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("Training coordinate labels contain NaN or Inf.")
    return df, x, y, feature_cols


def split_train_validation(x, y, seed=SEED, val_ratio=VAL_RATIO):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Training feature and label counts must match and be at least two.")
    train_idx, val_idx = train_test_split(
        np.arange(len(x)), test_size=val_ratio, shuffle=True,
        random_state=seed,
    )
    if len(set(train_idx) & set(val_idx)) > 0:
        raise AssertionError("Training and validation indices overlap.")
    return train_idx, val_idx


def load_final_test_dataset(path, feature_cols, label_cols=LABEL_COLS):
    df = read_and_sort_csv(path, require_file=False)
    missing_labels = [c for c in label_cols if c not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing coordinate columns in the test set: {missing_labels}")
    missing_ap = [c for c in feature_cols if c not in df.columns]
    if missing_ap:
        print(f"[WARN] The test set is missing {len(missing_ap)} AP columns; treating them as -110 dBm.")

    x = sanitize_rssi(df.reindex(columns=feature_cols).to_numpy(dtype=np.float32))
    y = df[list(label_cols)].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("Test coordinate labels contain NaN or Inf.")
    return df, x, y


def make_image_inputs(x):
    x = sanitize_rssi(x)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("AP data must be a two-dimensional matrix with at least one column.")
    n_samples, n_ap = x.shape
    side = math.ceil(math.sqrt(n_ap))
    unit = np.where(x != RSSI_MISSING, rssi_to_unit(x), 0.0)
    padded = np.zeros((n_samples, side * side), dtype=np.float32)
    padded[:, :n_ap] = unit
    return (padded.reshape(n_samples, side, side, 1) * 255.0).astype(np.float32)


def make_vector_inputs(x):
    x = sanitize_rssi(x)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("AP data must be a two-dimensional matrix with at least one column.")
    mask = (x != RSSI_MISSING).astype(np.float32)
    unit = np.where(mask > 0, rssi_to_unit(x), 0.0).astype(np.float32)
    return np.concatenate([unit, mask], axis=1).astype(np.float32)


def build_image_encoder(image_shape, embedding_dim=128):
    def conv_bn_relu(x, filters, stride, name):
        x = Conv2D(
            filters, kernel_size=3, strides=stride, padding="same",
            use_bias=False, kernel_regularizer=l2(1e-4),
            name=f"{name}_conv",
        )(x)
        x = BatchNormalization(name=f"{name}_bn")(x)
        return Activation("relu", name=f"{name}_relu")(x)

    image_in = Input(shape=image_shape, name="wifi_image_input")
    x = Rescaling(1.0 / 255.0, name="scale_01")(image_in)
    for filters, stride, name in (
        (32, 1, "stem1"), (32, 1, "stem2"),
        (64, 2, "s1_down"), (64, 1, "s1_1"),
        (128, 2, "s2_down"), (128, 1, "s2_1"),
        (256, 2, "s3_down"), (256, 1, "s3_1"),
    ):
        x = conv_bn_relu(x, filters, stride, name)
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization(name="wifi_bn_final")(x)
    x = Activation("relu", name="wifi_relu_final")(x)
    x = Dropout(0.4, name="wifi_dropout")(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4), name="img_fc1")(x)
    x = Dropout(0.1, name="img_fc1_dropout")(x)
    x = Dense(
        embedding_dim, activation="relu",
        kernel_regularizer=l2(1e-4), name="img_embedding",
    )(x)
    return Model(image_in, x, name="image_encoder")


@tf.keras.utils.register_keras_serializable(package="IPIN2016_Fusion")
class AddVectorTokens(tf.keras.layers.Layer):

    def __init__(self, n_ap, ap_id_embed_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.n_ap = int(n_ap)
        self.ap_id_embed_dim = int(ap_id_embed_dim)

    def build(self, input_shape):
        if input_shape[-1] != 2 * self.n_ap:
            raise ValueError("The vector input dimension must equal 2 * n_ap.")
        self.ap_id_embedding = self.add_weight(
            name="ap_id_embedding",
            shape=(self.n_ap, self.ap_id_embed_dim),
            initializer="glorot_uniform",
            regularizer=l2(1e-5),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        values = tf.expand_dims(x[:, :self.n_ap], axis=-1)
        mask = tf.expand_dims(x[:, self.n_ap:], axis=-1)
        embeddings = tf.tile(
            self.ap_id_embedding[None, :, :], [tf.shape(x)[0], 1, 1]
        )
        return tf.concat([values, mask, embeddings], axis=-1)

    def get_config(self):
        return {
            **super().get_config(),
            "n_ap": self.n_ap,
            "ap_id_embed_dim": self.ap_id_embed_dim,
        }


@tf.keras.utils.register_keras_serializable(package="IPIN2016_Fusion")
class ExtractObservationMask(tf.keras.layers.Layer):
    def call(self, tokens):
        return tokens[:, :, 1:2]


@tf.keras.utils.register_keras_serializable(package="IPIN2016_Fusion")
class MaskedAveragePooling(tf.keras.layers.Layer):
    def call(self, inputs):
        features, mask = inputs
        mask = tf.cast(mask, features.dtype)
        return (
            tf.reduce_sum(features * mask, axis=1)
            / (tf.reduce_sum(mask, axis=1) + 1e-6)
        )


@tf.keras.utils.register_keras_serializable(package="IPIN2016_Fusion")
class MaskedMaxPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        features, mask = inputs
        masked = tf.where(mask > 0, features, tf.ones_like(features) * -1e9)
        result = tf.reduce_max(masked, axis=1)
        has_observation = tf.reduce_any(mask > 0, axis=1)
        return tf.where(has_observation, result, tf.zeros_like(result))


@tf.keras.utils.register_keras_serializable(package="IPIN2016_Fusion")
class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=64, reg=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.reg = float(reg)
        self.fc1 = Dense(
            self.hidden_dim, activation="tanh",
            kernel_regularizer=l2(self.reg),
        )
        self.fc2 = Dense(1, activation=None, kernel_regularizer=l2(self.reg))

    def call(self, inputs):
        features, mask = inputs
        scores = self.fc2(self.fc1(features))
        scores = tf.where(mask > 0, scores, tf.ones_like(scores) * -1e9)
        attention = tf.nn.softmax(scores, axis=1)

        return tf.reduce_sum(attention * features, axis=1)

    def get_config(self):
        return {**super().get_config(), "hidden_dim": self.hidden_dim, "reg": self.reg}


@tf.keras.utils.register_keras_serializable(package="IPIN2016_Fusion")
class OneMinus(tf.keras.layers.Layer):
    def call(self, x):
        return 1.0 - x


@tf.keras.utils.register_keras_serializable(package="IPIN2016_Fusion")
class AbsoluteDifference(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs[0] - inputs[1])


def build_vector_encoder(n_ap, embedding_dim=128):
    vector_in = Input(shape=(2 * n_ap,), name="wifi_vector_input")
    tokens = AddVectorTokens(n_ap=n_ap, ap_id_embed_dim=16, name="vec_add_tokens")(vector_in)
    obs_mask = ExtractObservationMask(name="vec_obs_mask")(tokens)

    x = GaussianNoise(0.01, name="vec_token_noise")(tokens)
    for units, index in ((64, 1), (128, 2), (128, 3)):
        x = Dense(
            units, use_bias=False, kernel_regularizer=l2(1e-5),
            name=f"vec_token_fc{index}",
        )(x)
        x = BatchNormalization(name=f"vec_token_bn{index}")(x)
        x = Activation("relu", name=f"vec_token_relu{index}")(x)

    observed = Multiply(name="vec_apply_obs_mask")([x, obs_mask])
    avg = MaskedAveragePooling(name="vec_obs_masked_avg_pool")(
        [observed, obs_mask]
    )
    max_val = MaskedMaxPooling(name="vec_obs_masked_max_pool")(
        [observed, obs_mask]
    )
    attention = AttentionPooling(
        hidden_dim=64, reg=1e-5, name="vec_obs_att_pool"
    )([observed, obs_mask])
    pooled = Concatenate(name="vec_pool_concat")([avg, max_val, attention])

    x = Dense(256, use_bias=False, kernel_regularizer=l2(1e-5), name="vec_fuse_fc1")(pooled)
    x = BatchNormalization(name="vec_fuse_bn1")(x)
    x = Activation("relu", name="vec_fuse_relu1")(x)
    x = Dropout(0.20, name="vec_fuse_drop1")(x)
    x = Dense(128, use_bias=False, kernel_regularizer=l2(1e-5), name="vec_fuse_fc2")(x)
    x = BatchNormalization(name="vec_fuse_bn2")(x)
    x = Activation("relu", name="vec_fuse_relu2")(x)
    x = Dropout(0.15, name="vec_fuse_drop2")(x)
    x = Dense(
        embedding_dim, activation="relu",
        kernel_regularizer=l2(1e-5), name="vec_embedding",
    )(x)
    return Model(vector_in, x, name="vector_encoder")


def build_fusion_model(image_shape, n_ap, embedding_dim=128):
    img_encoder = build_image_encoder(image_shape, embedding_dim)
    vec_encoder = build_vector_encoder(n_ap, embedding_dim)
    img_feat, vec_feat = img_encoder.output, vec_encoder.output

    context = Concatenate(name="fusion_context")([img_feat, vec_feat])
    gate = Dense(
        embedding_dim, activation="relu",
        kernel_regularizer=l2(1e-4), name="fusion_gate_fc1",
    )(context)
    gate = Dropout(0.10, name="fusion_gate_drop1")(gate)
    gate = Dense(
        embedding_dim, activation="sigmoid",
        kernel_regularizer=l2(1e-4), name="fusion_gate",
    )(gate)

    img_weighted = Multiply(name="fusion_img_weighted")([gate, img_feat])
    vec_weighted = Multiply(name="fusion_vec_weighted")(
        [OneMinus(name="fusion_inv_gate")(gate), vec_feat]
    )
    weighted_sum = Add(name="fusion_weighted_sum")(
        [img_weighted, vec_weighted]
    )
    product = Multiply(name="fusion_feat_prod")([img_feat, vec_feat])
    difference = AbsoluteDifference(name="fusion_feat_absdiff")(
        [img_feat, vec_feat]
    )
    head = Concatenate(name="fusion_head_input")(
        [weighted_sum, product, difference]
    )
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4), name="fusion_fc1")(head)
    x = Dropout(0.20, name="fusion_drop1")(x)
    x = Dense(128, activation="relu", kernel_regularizer=l2(1e-4), name="fusion_fc2")(x)
    x = Dropout(0.10, name="fusion_drop2")(x)
    output = Dense(2, activation="linear", name="curr_output")(x)
    return Model(
        inputs=[img_encoder.input, vec_encoder.input],
        outputs=output,
        name="wifi_localization_fusion",
    )


def save_training_curves(history, plot_dir):
    for keys, filename, title in (
        (("loss", "val_loss"), "training_loss.png", "Train / Validation Loss"),
        (("mae", "val_mae"), "training_mae.png", "Train / Validation MAE"),
    ):
        if not all(k in history.history for k in keys):
            continue
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        for key in keys:
            ax.plot(history.history[key], label=key)
        ax.set(xlabel="Epoch", ylabel=keys[0], title=title)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, bbox_inches="tight")
        plt.close(fig)


def save_final_predictions(y_true, y_pred, test_df, output_dir, plot_dir):
    diff = y_pred - y_true
    errors = np.linalg.norm(diff, axis=1)
    result = pd.DataFrame({
        "fingerprint_index": np.arange(len(y_true), dtype=np.int32),
        "test_row_index": test_df["source_row_index"].to_numpy(dtype=np.int64),
        "file": test_df["file"].astype(str).to_numpy(),
        "Timestamp": test_df["Timestamp"].to_numpy(),
        "true_x_m": y_true[:, 0],
        "true_y_m": y_true[:, 1],
        "pred_x_m": y_pred[:, 0],
        "pred_y_m": y_pred[:, 1],
        "error_x_m": diff[:, 0],
        "error_y_m": diff[:, 1],
        "abs_error_x_m": np.abs(diff[:, 0]),
        "abs_error_y_m": np.abs(diff[:, 1]),
        "euclidean_error_m": errors,
    })
    result.to_csv(output_dir / "test_predictions.csv",
                  index=False, encoding="utf-8-sig")
    result.to_excel(output_dir / "test_predictions.xlsx", index=False)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    points = ax.scatter(
        y_true[:, 0], y_true[:, 1], c=errors, cmap="turbo",
        s=35, alpha=0.9, edgecolors="#233849", linewidths=0.2,
    )
    fig.colorbar(points, ax=ax, label="Localization Error (m)")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="Test Position Errors")
    ax.axis("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "test_error_scatter.png", bbox_inches="tight")
    plt.close(fig)
    return errors


def run_region(region, strategy):
    region_name = region["name"]
    out_dir = Path(OUTPUT_ROOT) / region_name
    plot_dir = out_dir / "plots"
    log_dir = out_dir / "logs"
    for folder in (out_dir, plot_dir, log_dir):
        folder.mkdir(parents=True, exist_ok=True)

    tf.keras.backend.clear_session()
    set_seed(SEED)
    df, x, y, feature_cols = load_training_dataset(region["train"])
    idx_train, idx_val = split_train_validation(x, y)
    x_train, x_val = x[idx_train], x[idx_val]
    y_train, y_val = y[idx_train], y[idx_val]

    print(
        f"[INFO] {region_name} | total={len(x)}, "
        f"train={len(idx_train)}, val={len(idx_val)}, AP={len(feature_cols)}"
    )

    df.iloc[idx_train].to_csv(
        out_dir / "train_80_percent.csv", index=False, encoding="utf-8-sig"
    )
    df.iloc[idx_val].to_csv(
        out_dir / "validation_20_percent.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({
        "source_row_index": np.concatenate([
            df.iloc[idx_train]["source_row_index"].to_numpy(),
            df.iloc[idx_val]["source_row_index"].to_numpy(),
        ]),
        "subset": ["train"] * len(idx_train) + ["validation"] * len(idx_val),
    }).sort_values("source_row_index").to_csv(
        out_dir / "train_val_split_indices.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.Series(feature_cols, name="ap_feature").to_csv(
        out_dir / "ap_feature_columns.csv", index=False, encoding="utf-8-sig"
    )

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train).astype(np.float32)
    y_val_scaled = scaler_y.transform(y_val).astype(np.float32)
    np.savez(
        out_dir / "target_scaler_y.npz",
        mean_=scaler_y.mean_, scale_=scaler_y.scale_, var_=scaler_y.var_,
    )

    train_inputs = [make_image_inputs(x_train), make_vector_inputs(x_train)]
    val_inputs = [make_image_inputs(x_val), make_vector_inputs(x_val)]

    with strategy.scope():
        model = build_fusion_model(
            image_shape=train_inputs[0].shape[1:],
            n_ap=len(feature_cols),
        )
        model.compile(
            optimizer=Adam(learning_rate=3e-4, clipnorm=0.3),
            loss="mse", metrics=["mae"],
        )
    model.summary()
    with (log_dir / "model_summary.txt").open("w", encoding="utf-8") as stream:
        with contextlib.redirect_stdout(stream):
            model.summary()

    best_weights_path = out_dir / "best_val_loss_fusion.weights.h5"
    callbacks = [
        EarlyStopping(
            monitor="val_loss", mode="min", patience=30,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5,
            patience=15, min_lr=1e-6, verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=True, verbose=1,
        ),
    ]


    history = model.fit(
        train_inputs, y_train_scaled,
        validation_data=(val_inputs, y_val_scaled),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1,
    )
    if not best_weights_path.is_file():
        raise FileNotFoundError(f"Best-validation weights file does not exist: {best_weights_path}")
    model.load_weights(str(best_weights_path))
    print(f"[INFO] {region_name}: loaded the weights with the lowest val_loss.")

    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_df.to_csv(out_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    hist_df.to_excel(out_dir / "training_history.xlsx", index=False)
    save_training_curves(history, plot_dir)


    model.save(str(out_dir / "fusion_final.keras"))
    model.save_weights(str(out_dir / "fusion_final.weights.h5"))


    test_df, x_test, y_test = load_final_test_dataset(
        region["test"], feature_cols
    )
    test_inputs = [make_image_inputs(x_test), make_vector_inputs(x_test)]
    test_preds_scaled = model.predict(
        test_inputs, batch_size=BATCH_SIZE, verbose=0
    )
    y_pred = scaler_y.inverse_transform(test_preds_scaled.astype(np.float64))
    y_test_scaled = scaler_y.transform(y_test).astype(np.float32)
    errors = save_final_predictions(
        y_test, y_pred, test_df, out_dir, plot_dir
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    report = {
        "region": region_name,
        "model_mode": "fusion",
        "seed": SEED,
        "train_samples": len(idx_train),
        "validation_samples": len(idx_val),
        "test_samples": len(y_test),
        "best_epoch_by_val_loss": best_epoch,
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "last_epoch_train_loss": float(history.history["loss"][-1]),
        "last_epoch_val_loss": float(history.history["val_loss"][-1]),

        "test_mse_scaled": float(np.mean((test_preds_scaled - y_test_scaled)**2)),
        "test_mae_scaled": float(np.mean(np.abs(test_preds_scaled - y_test_scaled))),
        "mean_position_error_m": float(np.mean(errors)),
        "median_position_error_m": float(np.median(errors)),
        "p75_position_error_m": float(np.percentile(errors, 75)),
        "p90_position_error_m": float(np.percentile(errors, 90)),
        "p95_position_error_m": float(np.percentile(errors, 95)),
        "max_position_error_m": float(np.max(errors)),
        "mae_x_m": float(np.mean(np.abs(y_pred[:, 0] - y_test[:, 0]))),
        "mae_y_m": float(np.mean(np.abs(y_pred[:, 1] - y_test[:, 1]))),
    }
    (out_dir / "metrics_report.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in report.items()) + "\n",
        encoding="utf-8",
    )
    print(f"[RESULT] {region_name}: mean error = {report['mean_position_error_m']:.4f} m")
    return report


def main():
    set_seed(SEED)
    strategy = configure_gpu()
    Path(OUTPUT_ROOT).mkdir(exist_ok=True, parents=True)
    all_reports = []
    for region in REGIONS:
        print(f"\n{'=' * 80}\n[INFO] Region: {region['name']}\n{'=' * 80}")
        try:
            report = run_region(region, strategy)
            all_reports.append(report)
        except Exception as exc:
            traceback.print_exc()
            print(f"[ERROR] {region['name']} failed: {exc}")
            all_reports.append({
                "region": region["name"],
                "model_mode": "fusion",
                "error": repr(exc),
            })

    summary = pd.DataFrame(all_reports)
    summary.to_csv(
        Path(OUTPUT_ROOT) / "fusion_all_regions_summary.csv",
        index=False, encoding="utf-8-sig",
    )
    summary.to_excel(
        Path(OUTPUT_ROOT) / "fusion_all_regions_summary.xlsx",
        index=False,
    )
    print("\n[INFO] All regions processed. Results saved to:", Path(OUTPUT_ROOT).resolve())


if __name__ == "__main__":
    main()
