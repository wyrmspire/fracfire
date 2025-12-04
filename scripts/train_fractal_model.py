
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from sklearn.model_selection import train_test_split

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def create_fractal_net(micro_shape, meso_shape, macro_shape, output_dim=5):
    """
    Multi-Input CNN (FractalNet).
    """
    # Branch 1: Micro (1m bars)
    input_micro = layers.Input(shape=micro_shape, name='micro_input')
    x1 = layers.Conv1D(32, 3, activation='relu')(input_micro)
    x1 = layers.MaxPooling1D(2)(x1)
    x1 = layers.Conv1D(64, 3, activation='relu')(x1)
    x1 = layers.MaxPooling1D(2)(x1)
    x1 = layers.Flatten()(x1)
    x1 = layers.Dense(64, activation='relu')(x1)
    
    # Branch 2: Meso (15m bars)
    input_meso = layers.Input(shape=meso_shape, name='meso_input')
    x2 = layers.Conv1D(32, 3, activation='relu')(input_meso)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = layers.Flatten()(x2)
    x2 = layers.Dense(32, activation='relu')(x2)
    
    # Branch 3: Macro (1h bars)
    input_macro = layers.Input(shape=macro_shape, name='macro_input')
    x3 = layers.Conv1D(32, 3, activation='relu')(input_macro)
    x3 = layers.MaxPooling1D(2)(x3)
    x3 = layers.Flatten()(x3)
    x3 = layers.Dense(32, activation='relu')(x3)
    
    # Fusion
    merged = layers.Concatenate()([x1, x2, x3])
    
    # Dense Layers
    z = layers.Dense(128, activation='relu')(merged)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(64, activation='relu')(z)
    
    # Output Head (Regression)
    # [Displacement, HighExc, LowExc, TotalDist, CloseLoc]
    outputs = layers.Dense(output_dim, activation='linear', name='output')(z)
    
    model = Model(inputs=[input_micro, input_meso, input_macro], outputs=outputs)
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    print("Loading Fractal Dataset...")
    data_path = "data/processed/fractal_dataset.npz"
    if not os.path.exists(data_path):
        print("Dataset not found.")
        return
        
    data = np.load(data_path)
    X_micro = data['X_micro']
    X_meso = data['X_meso']
    X_macro = data['X_macro']
    Y = data['Y_targets']
    
    print(f"Samples: {len(Y)}")
    
    # Split
    # We need to split all inputs
    indices = np.arange(len(Y))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=False)
    
    X_micro_train, X_micro_val = X_micro[train_idx], X_micro[val_idx]
    X_meso_train, X_meso_val = X_meso[train_idx], X_meso[val_idx]
    X_macro_train, X_macro_val = X_macro[train_idx], X_macro[val_idx]
    Y_train, Y_val = Y[train_idx], Y[val_idx]
    
    print("Building FractalNet...")
    model = create_fractal_net(
        micro_shape=X_micro.shape[1:],
        meso_shape=X_meso.shape[1:],
        macro_shape=X_macro.shape[1:]
    )
    
    model.summary()
    
    print("Training...")
    history = model.fit(
        [X_micro_train, X_meso_train, X_macro_train],
        Y_train,
        epochs=10, # Fast training
        batch_size=32,
        validation_data=([X_micro_val, X_meso_val, X_macro_val], Y_val),
        verbose=1
    )
    
    # Save
    model.save("out/fractal_net.keras")
    print("Saved model to out/fractal_net.keras")
    
    # Evaluate
    print("\n--- Evaluation (Validation Set) ---")
    preds = model.predict([X_micro_val, X_meso_val, X_macro_val])
    
    # Compare MAE for each target
    target_names = ['Displacement', 'HighExc', 'LowExc', 'TotalDist', 'CloseLoc']
    mae = np.mean(np.abs(preds - Y_val), axis=0)
    
    for i, name in enumerate(target_names):
        print(f"{name} MAE: {mae[i]:.4f}")
        
    # Example Prediction
    print("\nExample Prediction vs Actual:")
    print(f"Pred: {preds[0]}")
    print(f"Act : {Y_val[0]}")

if __name__ == "__main__":
    main()
