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
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense,
    Dropout, GaussianNoise, GlobalAveragePooling2D, Input, Multiply,
    Rescaling,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

SEED = 42
RSSI_MISSING = -100.0
RSSI_SCALE_MIN = -100.0
RSSI_SCALE_MAX = 0.0
VAL_RATIO = 0.20

TRAIN_FILE = "wifi_training.xlsx"
TEST_FILE = "wifi_test.xlsx"
LABEL_COLS = ("x_m", "y_m")
OUTPUT_DIR = "IPIN2018_DRG-FiLoc"
EPOCHS = 800
BATCH_SIZE = 64

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print(f"[WARN] GPU memory-growth configuration skipped: {exc}")

strategy = (
    tf.distribute.MirroredStrategy()
    if len(gpus) > 1 else tf.distribute.get_strategy()
)


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


def get_feature_columns(train_df, label_cols=LABEL_COLS):
    cols = list(train_df.columns)
    if len(cols) < 5:
        raise ValueError("The training table requires at least two metadata columns, one AP feature column, and two coordinate columns.")
    if any(col not in train_df.columns for col in label_cols):
        raise ValueError(f"Missing coordinate columns in the training table: {label_cols}")
    if "file" not in train_df.columns:
        raise ValueError("The training table must contain a file column.")
    feature_cols = [col for col in cols[2:-2] if col not in label_cols]
    if not feature_cols:
        raise ValueError("The training table contains no AP feature columns.")
    if len(feature_cols) != len(set(feature_cols)):
        raise ValueError("The training table contains duplicate AP column names.")
    return feature_cols


def load_training_file(train_path, label_cols=LABEL_COLS):
    df = pd.read_excel(train_path)
    feature_cols = get_feature_columns(df, label_cols)
    if len(df) < 2:
        raise ValueError("At least two samples are required to split the training file into training and validation sets.")
    x = sanitize_rssi(df[feature_cols].to_numpy(dtype=np.float32))
    y = df[list(label_cols)].to_numpy(dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("Training coordinate labels contain NaN or infinity.")
    return df, x, y, feature_cols


def split_training_validation(x, y, seed=SEED, val_ratio=VAL_RATIO):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("X and y must have matching sample counts of at least two.")
    idx_train, idx_val = train_test_split(
        np.arange(len(x)), test_size=val_ratio,
        random_state=seed, shuffle=True,
    )
    assert len(set(idx_train) & set(idx_val)) == 0
    assert len(idx_train) + len(idx_val) == len(x)
    return idx_train, idx_val


def load_test_file(test_path, feature_cols, label_cols=LABEL_COLS):
    df = pd.read_excel(test_path)
    missing_labels = [c for c in label_cols if c not in df.columns]
    if missing_labels:
        raise ValueError(f"Missing coordinate columns in the test table: {missing_labels}")
    if len(df) == 0:
        raise ValueError("The test file must not be empty.")

    x = sanitize_rssi(df.reindex(columns=feature_cols).to_numpy(dtype=np.float32))
    y = df[list(label_cols)].to_numpy(dtype=np.float32)
    if not np.isfinite(y).all():
        raise ValueError("Test coordinate labels contain NaN or infinity.")
    logs = (
        df["file"].astype(str).str.strip().to_numpy()
        if "file" in df.columns
        else np.repeat("EVA_SINGLE_LOG", len(df))
    )
    return x, y, logs


def make_image_inputs(x):
    x = sanitize_rssi(x)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("WiFi features must have shape [num_samples, num_aps], with num_aps >= 1.")
    n, n_ap = x.shape
    side = math.ceil(math.sqrt(n_ap))
    unit = rssi_to_unit(x)
    unit[x == RSSI_MISSING] = 0.0
    padded = np.zeros((n, side * side), dtype=np.float32)
    padded[:, :n_ap] = unit
    return (padded.reshape(n, side, side, 1) * 255.0).astype(np.float32)


def make_vector_inputs(x):
    x = sanitize_rssi(x)
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("WiFi features must have shape [num_samples, num_aps].")
    mask = (x != RSSI_MISSING).astype(np.float32)
    unit = np.where(mask > 0, rssi_to_unit(x), 0.0).astype(np.float32)
    return np.concatenate([unit, mask], axis=1).astype(np.float32)


def build_image_encoder(input_shape, embedding_dim=128):
    def conv_bn_relu(t, channels, stride, name):
        t = Conv2D(
            channels, 3, strides=stride, padding="same",
            use_bias=False, kernel_regularizer=l2(1e-4),
            name=f"{name}_conv",
        )(t)
        t = BatchNormalization(name=f"{name}_bn")(t)
        return Activation("relu", name=f"{name}_relu")(t)

    inp = Input(shape=input_shape, name="wifi_image_input")
    x = Rescaling(1.0 / 255.0, name="scale_01")(inp)
    x = conv_bn_relu(x, 32, 1, "stem1")
    x = conv_bn_relu(x, 32, 1, "stem2")
    x = conv_bn_relu(x, 64, 2, "s1_down")
    x = conv_bn_relu(x, 64, 1, "s1_1")
    x = conv_bn_relu(x, 128, 2, "s2_down")
    x = conv_bn_relu(x, 128, 1, "s2_1")
    x = conv_bn_relu(x, 256, 2, "s3_down")
    x = conv_bn_relu(x, 256, 1, "s3_1")
    x = GlobalAveragePooling2D(name="gap")(x)
    x = BatchNormalization(name="wifi_bn_final")(x)
    x = Activation("relu", name="wifi_relu_final")(x)
    x = Dropout(0.4, name="wifi_dropout")(x)
    x = Dense(256, activation="relu", kernel_regularizer=l2(1e-4), name="img_fc1")(x)
    x = Dropout(0.1, name="img_fc1_dropout")(x)
    out = Dense(
        embedding_dim, activation="relu",
        kernel_regularizer=l2(1e-4), name="img_embedding",
    )(x)
    return Model(inp, out, name="image_encoder")


@tf.keras.utils.register_keras_serializable(package="WiFiFusion")
class AddVectorTokens(tf.keras.layers.Layer):

    def __init__(self, n_ap, ap_id_embed_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.n_ap = int(n_ap)
        self.ap_id_embed_dim = int(ap_id_embed_dim)

    def build(self, input_shape):
        if input_shape[-1] != 2 * self.n_ap:
            raise ValueError("The vector input must contain n_ap RSSI values and n_ap mask values.")
        self.ap_id_embedding = self.add_weight(
            name="ap_id_embedding",
            shape=(self.n_ap, self.ap_id_embed_dim),
            initializer="glorot_uniform",
            regularizer=l2(1e-5),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x_unit = tf.expand_dims(inputs[:, :self.n_ap], -1)
        mask = tf.expand_dims(inputs[:, self.n_ap:], -1)
        batch_size = tf.shape(inputs)[0]
        ap_ids = tf.tile(tf.expand_dims(self.ap_id_embedding, 0), [batch_size, 1, 1])
        return tf.concat([x_unit, mask, ap_ids], axis=-1)

    def get_config(self):
        return {
            **super().get_config(),
            "n_ap": self.n_ap,
            "ap_id_embed_dim": self.ap_id_embed_dim,
        }


@tf.keras.utils.register_keras_serializable(package="WiFiFusion")
class ExtractObservationMask(tf.keras.layers.Layer):
    def call(self, tokens):
        return tokens[:, :, 1:2]


@tf.keras.utils.register_keras_serializable(package="WiFiFusion")
class MaskedGlobalAveragePooling1D(tf.keras.layers.Layer):
    def call(self, inputs):
        feats, mask = inputs
        mask = tf.cast(mask, feats.dtype)
        return (
            tf.reduce_sum(feats * mask, axis=1) /
            (tf.reduce_sum(mask, axis=1) + 1e-6)
        )


@tf.keras.utils.register_keras_serializable(package="WiFiFusion")
class MaskedGlobalMaxPooling1D(tf.keras.layers.Layer):
    def call(self, inputs):
        feats, mask = inputs
        masked = tf.where(mask > 0, feats, tf.ones_like(feats) * -1e9)
        result = tf.reduce_max(masked, axis=1)
        valid = tf.reduce_any(mask > 0, axis=1)
        return tf.where(valid, result, tf.zeros_like(result))


@tf.keras.utils.register_keras_serializable(package="WiFiFusion")
class AttentionPooling1D(tf.keras.layers.Layer):
    def __init__(self, hidden_dim=64, reg=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.reg = float(reg)
        self.fc1 = Dense(
            self.hidden_dim, activation="tanh",
            kernel_regularizer=l2(self.reg),
        )
        self.fc2 = Dense(1, kernel_regularizer=l2(self.reg))

    def call(self, inputs):
        feats, mask = inputs
        scores = self.fc2(self.fc1(feats))
        scores = tf.where(mask > 0, scores, tf.ones_like(scores) * -1e9)
        weights = tf.nn.softmax(scores, axis=1)

        return tf.reduce_sum(weights * feats, axis=1)

    def get_config(self):
        return {
            **super().get_config(),
            "hidden_dim": self.hidden_dim,
            "reg": self.reg,
        }


@tf.keras.utils.register_keras_serializable(package="WiFiFusion")
class OneMinus(tf.keras.layers.Layer):
    def call(self, inputs):
        return 1.0 - inputs


@tf.keras.utils.register_keras_serializable(package="WiFiFusion")
class AbsoluteDifference(tf.keras.layers.Layer):
    def call(self, inputs):
        left, right = inputs
        return tf.abs(left - right)


def build_vector_encoder(n_ap, embedding_dim=128, ap_id_embed_dim=16):
    inp = Input(shape=(2 * n_ap,), name="wifi_vector_input")
    tokens = AddVectorTokens(n_ap, ap_id_embed_dim, name="vec_add_tokens")(inp)
    obs_mask = ExtractObservationMask(name="vec_obs_mask")(tokens)

    x = GaussianNoise(0.01, name="vec_token_noise")(tokens)
    for channels, index in ((64, 1), (128, 2), (128, 3)):
        x = Dense(
            channels, use_bias=False,
            kernel_regularizer=l2(1e-5),
            name=f"vec_token_fc{index}",
        )(x)
        x = BatchNormalization(name=f"vec_token_bn{index}")(x)
        x = Activation("relu", name=f"vec_token_relu{index}")(x)

    x_obs = Multiply(name="vec_apply_obs_mask")([x, obs_mask])
    avg_pool = MaskedGlobalAveragePooling1D(name="vec_obs_masked_avg_pool")(
        [x_obs, obs_mask]
    )
    max_pool = MaskedGlobalMaxPooling1D(name="vec_obs_masked_max_pool")(
        [x_obs, obs_mask]
    )
    att_pool = AttentionPooling1D(
        hidden_dim=64, reg=1e-5, name="vec_obs_att_pool"
    )([x_obs, obs_mask])

    pooled = Concatenate(name="vec_pool_concat")([avg_pool, max_pool, att_pool])
    x = Dense(256, use_bias=False, kernel_regularizer=l2(1e-5), name="vec_fuse_fc1")(pooled)
    x = BatchNormalization(name="vec_fuse_bn1")(x)
    x = Activation("relu", name="vec_fuse_relu1")(x)
    x = Dropout(0.20, name="vec_fuse_drop1")(x)
    x = Dense(128, use_bias=False, kernel_regularizer=l2(1e-5), name="vec_fuse_fc2")(x)
    x = BatchNormalization(name="vec_fuse_bn2")(x)
    x = Activation("relu", name="vec_fuse_relu2")(x)
    x = Dropout(0.15, name="vec_fuse_drop2")(x)
    out = Dense(
        embedding_dim, activation="relu",
        kernel_regularizer=l2(1e-5), name="vec_embedding",
    )(x)
    return Model(inp, out, name="vector_encoder")


def build_fusion_model(image_shape, n_ap, output_dim=2, embedding_dim=128):
    image_encoder = build_image_encoder(image_shape, embedding_dim)
    vector_encoder = build_vector_encoder(n_ap, embedding_dim, ap_id_embed_dim=16)
    image_feat = image_encoder.output
    vector_feat = vector_encoder.output

    context = Concatenate(name="fusion_context")([image_feat, vector_feat])
    gate = Dense(
        embedding_dim, activation="relu",
        kernel_regularizer=l2(1e-4), name="fusion_gate_fc1",
    )(context)
    gate = Dropout(0.10, name="fusion_gate_drop1")(gate)
    gate = Dense(
        embedding_dim, activation="sigmoid",
        kernel_regularizer=l2(1e-4), name="fusion_gate",
    )(gate)

    image_weighted = Multiply(name="fusion_img_weighted")([gate, image_feat])
    vector_weighted = Multiply(name="fusion_vec_weighted")(
        [OneMinus(name="fusion_inv_gate")(gate), vector_feat]
    )
    weighted_sum = Add(name="fusion_weighted_sum")(
        [image_weighted, vector_weighted]
    )
    feature_product = Multiply(name="fusion_feat_prod")(
        [image_feat, vector_feat]
    )
    feature_absdiff = AbsoluteDifference(name="fusion_feat_absdiff")(
        [image_feat, vector_feat]
    )
    head_input = Concatenate(name="fusion_head_input")(
        [weighted_sum, feature_product, feature_absdiff]
    )
    x = Dense(
        256, activation="relu",
        kernel_regularizer=l2(1e-4), name="fusion_fc1",
    )(head_input)
    x = Dropout(0.20, name="fusion_drop1")(x)
    x = Dense(
        128, activation="relu",
        kernel_regularizer=l2(1e-4), name="fusion_fc2",
    )(x)
    x = Dropout(0.10, name="fusion_drop2")(x)
    prediction = Dense(output_dim, activation="linear", name="curr_output")(x)
    return Model(
        inputs=[image_encoder.input, vector_encoder.input],
        outputs=prediction,
        name="wifi_localization_fusion",
    )


def plot_training_curves(history, output_dir):
    h = history.history
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.plot(h["loss"], label="Train loss")
    ax.plot(h["val_loss"], label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Fusion Training and Validation Loss")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss.png")
    plt.close(fig)
    if "mae" in h and "val_mae" in h:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        ax.plot(h["mae"], label="Train MAE")
        ax.plot(h["val_mae"], label="Validation MAE")
        ax.set(xlabel="Epoch", ylabel="Scaled MAE", title="Fusion MAE")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_mae.png")
        plt.close(fig)


def save_test_results(y_true, y_pred, logs, output_dir):
    delta = y_pred - y_true
    errors = np.linalg.norm(delta, axis=1)
    table = pd.DataFrame({
        "test_row_index": np.arange(len(y_true), dtype=np.int32),
        "file": logs,
        "true_x_m": y_true[:, 0],
        "true_y_m": y_true[:, 1],
        "pred_x_m": y_pred[:, 0],
        "pred_y_m": y_pred[:, 1],
        "error_x_m": delta[:, 0],
        "error_y_m": delta[:, 1],
        "abs_error_x_m": np.abs(delta[:, 0]),
        "abs_error_y_m": np.abs(delta[:, 1]),
        "euclidean_error_m": errors,
    })
    table.to_csv(output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    table.to_excel(output_dir / "test_predictions.xlsx", index=False)

    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    dots = ax.scatter(
        y_true[:, 0], y_true[:, 1],
        c=errors, cmap="turbo", s=35, alpha=0.9,
        edgecolors="#233849", linewidths=0.2,
    )
    fig.colorbar(dots, ax=ax, label="Localization Error (m)")
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="Test Position Errors")
    ax.axis("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "test_position_errors.png")
    plt.close(fig)
    return errors


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, x_all, y_all, feature_cols = load_training_file(TRAIN_FILE)
    train_idx, val_idx = split_training_validation(x_all, y_all)

    print(
        f"[INFO] Dataset split: total={len(x_all)}, "
        f"train={len(train_idx)}, val={len(val_idx)}, "
        f"AP={len(feature_cols)}, seed={SEED}"
    )

    split_df = pd.DataFrame({
        "row_index": np.concatenate([train_idx, val_idx]),
        "subset": (
            ["train"] * len(train_idx) +
            ["validation"] * len(val_idx)
        ),
    }).sort_values("row_index")
    split_df.to_csv(output_dir / "train_validation_split.csv", index=False)

    train_df.iloc[train_idx].to_excel(
        output_dir / "train_80_percent.xlsx", index=False
    )
    train_df.iloc[val_idx].to_excel(
        output_dir / "validation_20_percent.xlsx", index=False
    )

    x_train, x_val = x_all[train_idx], x_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]


    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train).astype(np.float32)
    y_val_s = scaler_y.transform(y_val).astype(np.float32)
    np.savez(
        output_dir / "target_scaler_y.npz",
        mean_=scaler_y.mean_,
        scale_=scaler_y.scale_,
        var_=scaler_y.var_,
    )
    pd.Series(feature_cols, name="ap_feature").to_csv(
        output_dir / "ap_feature_columns.csv", index=False, encoding="utf-8-sig"
    )

    train_inputs = [make_image_inputs(x_train), make_vector_inputs(x_train)]
    val_inputs = [make_image_inputs(x_val), make_vector_inputs(x_val)]
    image_shape = train_inputs[0].shape[1:]
    n_ap = x_train.shape[1]
    print(f"[INFO] Image input: {train_inputs[0].shape}")
    print(f"[INFO] Vector input: {train_inputs[1].shape}")

    with strategy.scope():
        model = build_fusion_model(
            image_shape=image_shape, n_ap=n_ap, output_dim=y_train.shape[1]
        )
        model.compile(
            optimizer=Adam(learning_rate=3e-4, clipnorm=0.3),
            loss="mse", metrics=["mae"],
        )

    with (output_dir / "model_summary.txt").open("w", encoding="utf-8") as file:
        with contextlib.redirect_stdout(file):
            model.summary()
    model.summary()

    best_weights_path = output_dir / "best_val_loss_fusion.weights.h5"
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
            filepath=str(best_weights_path), monitor="val_loss",
            mode="min", save_best_only=True,
            save_weights_only=True, verbose=1,
        ),
    ]


    history = model.fit(
        train_inputs, y_train_s,
        validation_data=(val_inputs, y_val_s),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=callbacks, verbose=1,
    )
    if not best_weights_path.exists():
        raise FileNotFoundError(f"Training finished, but best-validation weights are missing: {best_weights_path}")
    model.load_weights(str(best_weights_path))
    print(f"[INFO] Loaded best validation-loss weights: {best_weights_path}")

    pd.DataFrame(history.history).to_csv(
        output_dir / "training_history.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(history.history).to_excel(
        output_dir / "training_history.xlsx", index=False
    )
    plot_training_curves(history, output_dir)


    model.save(str(output_dir / "fusion_final.keras"))
    model.save_weights(str(output_dir / "fusion_final.weights.h5"))


    x_test, y_test, test_logs = load_test_file(TEST_FILE, feature_cols)
    test_inputs = [make_image_inputs(x_test), make_vector_inputs(x_test)]
    y_pred_s = model.predict(test_inputs, batch_size=BATCH_SIZE, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_s)
    errors = save_test_results(y_test, y_pred, test_logs, output_dir)

    test_mse_scaled = float(np.mean(np.square(
        scaler_y.transform(y_test).astype(np.float32) - y_pred_s
    )))
    test_mae_scaled = float(np.mean(np.abs(
        scaler_y.transform(y_test).astype(np.float32) - y_pred_s
    )))
    val_losses = np.asarray(history.history["val_loss"])
    best_epoch = int(np.argmin(val_losses)) + 1
    metrics = {
        "model": "fusion",
        "random_seed": SEED,
        "train_samples": len(train_idx),
        "validation_samples": len(val_idx),
        "test_samples": len(y_test),
        "best_epoch": best_epoch,
        "best_val_loss": float(np.min(val_losses)),
        "final_epoch_train_loss": float(history.history["loss"][-1]),
        "final_epoch_val_loss": float(history.history["val_loss"][-1]),
        "test_mse_scaled": test_mse_scaled,
        "test_mae_scaled": test_mae_scaled,
        "mean_position_error_m": float(np.mean(errors)),
        "median_position_error_m": float(np.median(errors)),
        "p75_position_error_m": float(np.percentile(errors, 75)),
        "p90_position_error_m": float(np.percentile(errors, 90)),
        "p95_position_error_m": float(np.percentile(errors, 95)),
        "max_position_error_m": float(np.max(errors)),
        "mae_x_m": float(mean_absolute_error(y_test[:, 0], y_pred[:, 0])),
        "mae_y_m": float(mean_absolute_error(y_test[:, 1], y_pred[:, 1])),
    }
    report = "\n".join(f"{key}: {value}" for key, value in metrics.items())
    (output_dir / "metrics_report.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + "=" * 70)
    print(report)
    print(f"[INFO] All artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
