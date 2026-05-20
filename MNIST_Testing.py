import numpy as np
import os
import math
import sys

# Enable UTF-8 character support in Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

# Settings
DATA_FOLDER = ".\\FPGA_Final_Source Files\\test_files"

def load_decimal_txt(filename):
    """Reads decimal files similar to Verilog outputs."""
    path = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Did you run the Verilog simulation?")
        return None

    with open(path, 'r') as f:
        data = [int(line.strip()) for line in f if line.strip()]

    return np.array(data, dtype=np.int32)


# --- HELPER FUNCTIONS ---
def load_txt_to_array(filename, shape=None):
    path = os.path.join(DATA_FOLDER, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found!")

    with open(path, 'r') as f:
        data = [int(line.strip()) for line in f]

    arr = np.array(data, dtype=np.int32)

    if shape:
        arr = arr.reshape(shape)

    return arr


# --- FPGA HARDWARE FUNCTIONS (BIT SHIFTING) ---

def fpga_relu(x):
    return np.maximum(x, 0)


def fpga_fixed_quantize(x, shift_amount):
    """
    FPGA: Arithmetic Right Shift (>>>)
    Uses bit shifting instead of division.
    Example: shift_amount=8 divides the value by 256.
    """

    # 1. Bit shifting (right shift)
    # In Python, >> performs arithmetic bit shifting on integers.
    x_shifted = x >> shift_amount

    # 2. Clamping (saturation)
    # Clamp results to 8-bit limits (-128..127).
    # Since ReLU is used, lower bound is 0 and upper bound is 127.
    x_clamped = np.clip(x_shifted, 0, 127)

    return x_clamped.astype(np.int8)


# --- LAYERS ---
def layer_conv2d(img, w, b):
    h_in, w_in = img.shape
    _, _, _, n_filt = w.shape

    h_out, w_out = h_in - 2, w_in - 2

    output = np.zeros((h_out, w_out, n_filt), dtype=np.int32)

    for f in range(n_filt):
        wf = w[:, :, 0, f]
        bf = b[f]

        for r in range(h_out):
            for c in range(w_out):
                acc = np.sum(img[r:r+3, c:c+3] * wf) + bf
                output[r, c, f] = acc

    return output


def layer_maxpool(img):
    h, w, c = img.shape

    out = np.zeros((h//2, w//2, c), dtype=np.int32)

    for k in range(c):
        for r in range(h//2):
            for c_idx in range(w//2):
                out[r, c_idx, k] = np.max(
                    img[r*2:r*2+2, c_idx*2:c_idx*2+2, k]
                )

    return out


def layer_dense(flat_input, w, b):
    return np.dot(flat_input, w) + b


# --- CALIBRATION AND TEST LOGIC ---

def calculate_shift(max_val):
    """
    Calculates how many bits are needed to shift
    a value into the 0-127 range.
    """

    if max_val <= 127:
        return 0

    # log2(max_val) gives the number of bits required.
    # We want to keep values within 7 bits (0-127).
    # Example:
    # max=32000 (~15 bits) -> 15 - 7 = 8 bit shift
    bits_needed = math.ceil(math.log2(max_val))

    shift = bits_needed - 7

    return max(0, shift)


def run_fpga_pipeline():

    print("--- 1. LOADING WEIGHTS ---")

    w_conv = load_txt_to_array("conv1_weights.txt", (3,3,1,4))
    b_conv = load_txt_to_array("conv1_bias.txt")

    w_fc1  = load_txt_to_array("dense1_weights.txt", (676, 32))
    b_fc1  = load_txt_to_array("dense1_bias.txt")

    w_fc2  = load_txt_to_array("dense2_weights.txt", (32, 32))
    b_fc2  = load_txt_to_array("dense2_bias.txt")

    w_out  = load_txt_to_array("output_weights.txt", (32, 10))
    b_out  = load_txt_to_array("output_bias.txt")

    # Find test files
    all_files = sorted([
        f for f in os.listdir(DATA_FOLDER)
        if f.startswith("test_image")
    ])

    if not all_files:
        print("No test files found!")
        return

    # ---------------------------------------------------------
    # STEP 1: CALIBRATION (Find Shift Values)
    # ---------------------------------------------------------

    print("\n--- 2. CALIBRATION: CALCULATING OPTIMAL SHIFT VALUES ---")
    print("(Using the first 10 images to measure maximum layer outputs)")

    max_conv_acc = 0
    max_dense1_acc = 0
    max_dense2_acc = 0

    calibration_files = all_files[:10]

    for fname in calibration_files:

        img = load_txt_to_array(fname, (28, 28))

        # Conv1 calculation (raw accumulator)
        c1 = layer_conv2d(img, w_conv, b_conv)
        c1 = fpga_relu(c1)

        max_conv_acc = max(max_conv_acc, np.max(c1))

        # Temporary scaling for simulation
        c1_temp = (c1 >> 8).astype(np.int8)

        p1 = layer_maxpool(c1_temp)
        flat = p1.flatten()

        # Dense1 calculation
        d1 = layer_dense(flat, w_fc1, b_fc1)
        d1 = fpga_relu(d1)

        max_dense1_acc = max(max_dense1_acc, np.max(d1))

        d1_temp = (d1 >> 8).astype(np.int8)

        # Dense2 calculation
        d2 = layer_dense(d1_temp, w_fc2, b_fc2)
        d2 = fpga_relu(d2)

        max_dense2_acc = max(max_dense2_acc, np.max(d2))

    # Calculate shift values
    SHIFT_CONV   = calculate_shift(max_conv_acc)
    SHIFT_DENSE1 = calculate_shift(max_dense1_acc)
    SHIFT_DENSE2 = calculate_shift(max_dense2_acc)

    print("-" * 40)
    print("MAXIMUM VALUES AND REQUIRED SHIFTS:")
    print(f"Conv1 Max Acc  : {max_conv_acc}  -> Required Shift: {SHIFT_CONV}")
    print(f"Dense1 Max Acc : {max_dense1_acc}  -> Required Shift: {SHIFT_DENSE1}")
    print(f"Dense2 Max Acc : {max_dense2_acc}  -> Required Shift: {SHIFT_DENSE2}")
    print("-" * 40)

    print("\n[COPY FOR VERILOG]")
    print(f"parameter SHIFT_CONV   = {SHIFT_CONV};")
    print(f"parameter SHIFT_DENSE1 = {SHIFT_DENSE1};")
    print(f"parameter SHIFT_DENSE2 = {SHIFT_DENSE2};")
    print("-" * 40)

    # ---------------------------------------------------------
    # STEP 2: TESTING WITH FIXED SHIFT VALUES
    # ---------------------------------------------------------

    print("\n--- 3. TESTING WITH FIXED SHIFT VALUES ---")

    correct = 0

    for fname in all_files:

        real_label = int(
            fname.split("_label_")[1].split(".txt")[0]
        )

        img = load_txt_to_array(fname, (28, 28))

        # LAYER 1
        x = layer_conv2d(img, w_conv, b_conv)
        x = fpga_relu(x)
        x = fpga_fixed_quantize(x, SHIFT_CONV)

        # LAYER 2
        x = layer_maxpool(x)
        x = x.flatten()

        # LAYER 3
        x = layer_dense(x, w_fc1, b_fc1)
        x = fpga_relu(x)
        x = fpga_fixed_quantize(x, SHIFT_DENSE1)

        # LAYER 4
        x = layer_dense(x, w_fc2, b_fc2)
        x = fpga_relu(x)
        x = fpga_fixed_quantize(x, SHIFT_DENSE2)

        # OUTPUT LAYER
        # No shifting, only argmax is used
        final_scores = layer_dense(x, w_out, b_out)

        prediction = np.argmax(final_scores)

        is_ok = (prediction == real_label)

        if is_ok:
            correct += 1

        mark = "✅" if is_ok else "❌"

        print(f"{fname} -> Prediction: {prediction} {mark}")

    print(f"\nRESULT: {correct} correct out of {len(all_files)} tests.")
    print(f"Accuracy: %{(correct / len(all_files)) * 100:.1f}")


if __name__ == "__main__":
    run_fpga_pipeline()