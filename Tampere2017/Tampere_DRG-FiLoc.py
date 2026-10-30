import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import contextlib
import math
import random
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
    Input, Dense, BatchNormalization, Activation, GlobalAveragePooling2D,
    Dropout, Conv2D, Multiply, GaussianNoise, Concatenate, Add, Rescaling,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


TRAIN_FILE = "tampere_trainging_dataset.csv"
TEST_FILE = "tampere_test_dataset.csv"
LABEL_COLS = ("X", "Y")
N_FEATURE_COLS = 992

SEED = 42
VALIDATION_RATIO = 0.2
EPOCHS = 800
BATCH_SIZE = 64
OUTPUT_DIR = "Tampere_DRG-FiLoc"

RSSI_MISSING = -100.0
RSSI_SCALE_MIN = -100.0
RSSI_SCALE_MAX = 0.0

plt.rcParams["axes.unicode_minus"] = False


def set_global_seed(seed=SEED):
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
            print(f"[WARN] GPU memory growth not changed: {exc}")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
    print(f"[INFO] GPU count: {len(gpus)}; replicas: {strategy.num_replicas_in_sync}")
    return strategy


def read_table_auto(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data file format: {path}")


def sanitize_rssi(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(
        x, nan=RSSI_MISSING,
        posinf=RSSI_MISSING, neginf=RSSI_MISSING,
    )
    return x.astype(np.float32)


def rssi_to_unit(x):
    x = sanitize_rssi(x)
    z = (x - RSSI_SCALE_MIN) / (RSSI_SCALE_MAX - RSSI_SCALE_MIN)
    return np.clip(z, 0.0, 1.0).astype(np.float32)


def load_training_dataset(path, n_features=N_FEATURE_COLS, labels=LABEL_COLS):
    df = read_table_auto(path)
    if len(df) < 2:
        raise ValueError("The training set must contain at least two rows for a random split.")
    if df.shape[1] < n_features + len(labels):
        raise ValueError("Insufficient training columns: at least 992 AP columns and 2 coordinate columns are required.")
    if not set(labels).issubset(df.columns):
        raise ValueError(f"Missing coordinate columns in the training set: {labels}")
    feature_cols = list(df.columns[:n_features])
    if len(set(feature_cols)) != n_features:
        raise ValueError("The first 992 AP column names must be unique.")
    if set(feature_cols) & set(labels):
        raise ValueError("Coordinate columns must not be included among the AP feature columns.")

    x = sanitize_rssi(df[feature_cols].to_numpy(dtype=np.float32))

    y = df[list(labels)].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("The training data contains non-finite coordinate labels.")
    return df, x, y, feature_cols


def random_train_val_split(x, y, seed=SEED, validation_ratio=VALIDATION_RATIO):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Input and label sample counts must match and be at least two.")
    train_idx, val_idx = train_test_split(
        np.arange(len(x)), test_size=validation_ratio,
        random_state=seed, shuffle=True,
    )
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(train_idx) + len(val_idx) == len(x)
    return train_idx, val_idx


def load_test_dataset(path, feature_cols, labels=LABEL_COLS):
    df = read_table_auto(path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing AP feature columns in the test set: {missing[:10]}; {len(missing)} columns in total.")
    missing_labels = [c for c in labels if c not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing coordinate columns in the test set: {missing_labels}")
    if len(df) == 0:
        raise ValueError("The independent test set is empty.")

    x = sanitize_rssi(df[feature_cols].to_numpy(dtype=np.float32))
    y = df[list(labels)].to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("The test data contains non-finite coordinate labels.")
    return x, y


def make_image_inputs(x):
    x = sanitize_rssi(x)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("X must be a two-dimensional matrix containing at least one AP.")
    n, n_ap = x.shape
    side = int(math.ceil(math.sqrt(n_ap)))
    mask = (x != RSSI_MISSING)
    unit = np.where(mask, rssi_to_unit(x), 0.0)
    padded = np.zeros((n, side * side), dtype=np.float32)
    padded[:, :n_ap] = unit
    return (padded.reshape(n, side, side, 1) * 255.0).astype(np.float32)


def make_vector_inputs(x):
    x = sanitize_rssi(x)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("X must be a two-dimensional matrix containing at least one AP.")
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


@tf.keras.utils.register_keras_serializable(package="Tampere_Fusion")
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


@tf.keras.utils.register_keras_serializable(package="Tampere_Fusion")
class ExtractObservationMask(tf.keras.layers.Layer):
    def call(self, tokens):
        return tokens[:, :, 1:2]


@tf.keras.utils.register_keras_serializable(package="Tampere_Fusion")
class MaskedAveragePooling(tf.keras.layers.Layer):
    def call(self, inputs):
        features, mask = inputs
        mask = tf.cast(mask, features.dtype)
        return (
            tf.reduce_sum(features * mask, axis=1)
            / (tf.reduce_sum(mask, axis=1) + 1e-6)
        )


@tf.keras.utils.register_keras_serializable(package="Tampere_Fusion")
class MaskedMaxPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        features, mask = inputs
        masked = tf.where(mask > 0, features, tf.ones_like(features) * -1e9)
        result = tf.reduce_max(masked, axis=1)
        has_observation = tf.reduce_any(mask > 0, axis=1)
        return tf.where(has_observation, result, tf.zeros_like(result))


@tf.keras.utils.register_keras_serializable(package="Tampere_Fusion")
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


@tf.keras.utils.register_keras_serializable(package="Tampere_Fusion")
class OneMinus(tf.keras.layers.Layer):
    def call(self, x):
        return 1.0 - x


@tf.keras.utils.register_keras_serializable(package="Tampere_Fusion")
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


def plot_training_curves(history, plot_dir):
    data = history.history
    for name, columns, ylabel in (
        ("training_loss.png", ("loss", "val_loss"), "Loss (scaled)"),
        ("training_mae.png", ("mae", "val_mae"), "MAE (scaled)"),
    ):
        if not all(col in data for col in columns):
            continue
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        for col in columns:
            ax.plot(data[col], label=col)
        ax.set(xlabel="Epoch", ylabel=ylabel, title="Fusion Training History")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / name, bbox_inches="tight")
        plt.close(fig)


def save_test_results(y_true, y_pred, seed, output_dir, plot_dir):
    diff = y_pred - y_true
    absolute_error = np.abs(diff)
    error_norm = np.linalg.norm(diff, axis=1)
    predictions = pd.DataFrame({
        "seed": seed,
        "fingerprint_index": np.arange(len(y_true)),
        "true_x_m": y_true[:, 0],
        "true_y_m": y_true[:, 1],
        "pred_x_m": y_pred[:, 0],
        "pred_y_m": y_pred[:, 1],
        "error_x_m": diff[:, 0],
        "error_y_m": diff[:, 1],
        "abs_error_x_m": absolute_error[:, 0],
        "abs_error_y_m": absolute_error[:, 1],
        "euclidean_error": error_norm,
    })
    predictions.to_csv(
        output_dir / "prediction_results_fusion.csv",
        index=False, encoding="utf-8-sig",
    )
    predictions.to_excel(output_dir / "prediction_results_fusion.xlsx", index=False)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    dots = ax.scatter(
        y_true[:, 0], y_true[:, 1], c=error_norm, s=35,
        cmap="turbo", alpha=0.9, edgecolors="#253b4a", linewidths=0.2,
    )
    fig.colorbar(dots, ax=ax, label="Position Error (m)")
    ax.set(
        xlabel="X (m)", ylabel="Y (m)",
        title="Final Test Localization Error",
    )
    ax.axis("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "test_error_scatter.png", bbox_inches="tight")
    plt.close(fig)
    return error_norm


def main():
    set_global_seed(SEED)
    strategy = configure_gpu()

    root_dir = Path(OUTPUT_DIR)
    model_dir = root_dir / f"seed_{SEED}" / "fusion"
    plot_dir = model_dir / "plots"
    log_dir = model_dir / "logs"
    for folder in (model_dir, plot_dir, log_dir):
        folder.mkdir(parents=True, exist_ok=True)


    df, x_all, y_all, feature_cols = load_training_dataset(TRAIN_FILE)
    train_idx, val_idx = random_train_val_split(x_all, y_all)
    x_train, x_val = x_all[train_idx], x_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    print(
        f"[INFO] Total={len(x_all)} | Train={len(train_idx)} | "
        f"Validation={len(val_idx)} | APs={len(feature_cols)} | Seed={SEED}"
    )
    split_info = pd.DataFrame({
        "source_row_index": np.concatenate([train_idx, val_idx]),
        "subset": (
            ["train"] * len(train_idx) +
            ["validation"] * len(val_idx)
        ),
    }).sort_values("source_row_index")
    split_info.to_csv(model_dir / "train_val_indices.csv", index=False)
    df.iloc[train_idx].to_csv(
        model_dir / "train_80_percent.csv", index=False, encoding="utf-8-sig"
    )
    df.iloc[val_idx].to_csv(
        model_dir / "validation_20_percent.csv", index=False, encoding="utf-8-sig"
    )
    pd.Series(feature_cols, name="ap_feature").to_csv(
        model_dir / "ap_feature_columns.csv", index=False, encoding="utf-8-sig"
    )


    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train).astype(np.float32)
    y_val_scaled = scaler_y.transform(y_val).astype(np.float32)
    np.savez(
        model_dir / "target_scaler_y.npz",
        mean_=scaler_y.mean_, scale_=scaler_y.scale_, var_=scaler_y.var_,
    )

    train_inputs = [make_image_inputs(x_train), make_vector_inputs(x_train)]
    val_inputs = [make_image_inputs(x_val), make_vector_inputs(x_val)]
    print(f"[INFO] Image input: {train_inputs[0].shape}")
    print(f"[INFO] Vector input: {train_inputs[1].shape}")

    with strategy.scope():
        model = build_fusion_model(
            image_shape=train_inputs[0].shape[1:],
            n_ap=len(feature_cols),
        )
        model.compile(
            optimizer=Adam(learning_rate=3e-4, clipnorm=0.3),
            loss="mse",
            metrics=["mae"],
        )

    model.summary()
    with (log_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle):
            model.summary()

    best_weights_path = model_dir / "best_val_loss_fusion.weights.h5"
    callbacks = [
        EarlyStopping(
            monitor="val_loss", mode="min",
            patience=30, restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=15, min_lr=1e-6, verbose=1,
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
        raise FileNotFoundError(f"No weights checkpoint saved based on val_loss was found: {best_weights_path}")
    model.load_weights(str(best_weights_path))

    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(model_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    history_df.to_excel(model_dir / "training_history.xlsx", index=False)
    plot_training_curves(history, plot_dir)


    model.save(str(model_dir / "fusion_final.keras"))
    model.save_weights(str(model_dir / "fusion_final.weights.h5"))


    x_test, y_test = load_test_dataset(TEST_FILE, feature_cols)
    test_inputs = [make_image_inputs(x_test), make_vector_inputs(x_test)]
    y_test_scaled = scaler_y.transform(y_test).astype(np.float32)
    test_pred_scaled = model.predict(test_inputs, batch_size=BATCH_SIZE, verbose=0)
    y_pred = scaler_y.inverse_transform(test_pred_scaled.astype(np.float64))


    test_mse_scaled = float(np.mean(np.square(y_test_scaled - test_pred_scaled)))
    test_mae_scaled = float(np.mean(np.abs(y_test_scaled - test_pred_scaled)))
    error_norm = save_test_results(
        y_test, y_pred, SEED, model_dir, plot_dir
    )
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    metrics = {
        "model_mode": "fusion",
        "seed": SEED,
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(y_test),
        "best_epoch_by_val_loss": best_epoch,
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "test_mse_scaled": test_mse_scaled,
        "test_mae_scaled": test_mae_scaled,
        "mean_position_error": float(np.mean(error_norm)),
        "median_position_error": float(np.median(error_norm)),
        "p75_position_error": float(np.percentile(error_norm, 75)),
        "p90_position_error": float(np.percentile(error_norm, 90)),
        "p95_position_error": float(np.percentile(error_norm, 95)),
        "max_position_error": float(np.max(error_norm)),
        "mae_x_m": float(np.mean(np.abs(y_pred[:, 0] - y_test[:, 0]))),
        "mae_y_m": float(np.mean(np.abs(y_pred[:, 1] - y_test[:, 1]))),
    }
    report = "\n".join(f"{key}: {value}" for key, value in metrics.items())
    (model_dir / "metrics_report.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print(f"\n[INFO] Fusion output directory: {model_dir.resolve()}")


if __name__ == "__main__":
    main()
