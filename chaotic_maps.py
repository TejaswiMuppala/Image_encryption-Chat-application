import cv2
import numpy as np
import os

# Logistic Map Function for Chaos-Based Scrambling
def logistic_map(x, r, n):
    sequence = []
    for _ in range(n):
        x = r * x * (1 - x)
        sequence.append(x)
    return sequence

# Scramble Image Pixels Using Chaos
def scramble_pixels(image):
    h, w, c = image.shape
    total_pixels = h * w

    # Generate chaotic sequence
    x0 = 0.5   # Initial value
    r = 3.99   # Control parameter (close to 4 for high chaos)
    chaotic_sequence = logistic_map(x0, r, total_pixels)

    # Create scrambling indices based on chaotic values
    indices = np.argsort(chaotic_sequence)

    # Flatten image channels
    flat_img = image.reshape(-1, c)

    # Scramble using the indices
    scrambled_img = flat_img[indices].reshape(h, w, c)

    return scrambled_img, indices

# Unscramble Image Pixels Using Chaos
def unscramble_pixels(scrambled_img, indices):
    h, w, c = scrambled_img.shape
    total_pixels = h * w

    # Create an array to restore order
    restored = np.zeros_like(scrambled_img.reshape(-1, c))

    # Restore using original indices
    restored[indices] = scrambled_img.reshape(-1, c)

    return restored.reshape(h, w, c)

# Encrypt Image with Chaotic Scrambling
def encrypt_image_with_chaos(image_path, encrypted_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: '{image_path}' not found! Check the file path.")

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Failed to load '{image_path}'. Ensure it's a valid image format.")

    scrambled, indices = scramble_pixels(image)
    cv2.imwrite(encrypted_path, scrambled)
    print(f"[INFO] Encrypted (Scrambled) image saved as: {encrypted_path}")

    return indices

# Decrypt Image with Chaotic Unscrambling
def decrypt_image_with_chaos(encrypted_path, decrypted_path, indices):
    if not os.path.exists(encrypted_path):
        raise FileNotFoundError(f"Error: '{encrypted_path}' not found! Check the file path.")

    scrambled_img = cv2.imread(encrypted_path)
    if scrambled_img is None:
        raise ValueError(f"Error: Failed to load '{encrypted_path}'. Ensure it's a valid image format.")

    restored = unscramble_pixels(scrambled_img, indices)
    cv2.imwrite(decrypted_path, restored)
    print(f"[INFO] Decrypted (Restored) image saved as: {decrypted_path}")

# Main Execution
if __name__ == "__main__":
    image_path = input("Enter path of the image to encrypt: ").strip()
    encrypted_path = "chaos_encrypted.jpg"
    decrypted_path = "chaos_decrypted.jpg"

    print("[INFO] Encrypting Image with Chaotic Maps...")
    indices = encrypt_image_with_chaos(image_path, encrypted_path)

    print("[INFO] Decrypting Image with Chaotic Maps...")
    decrypt_image_with_chaos(encrypted_path, decrypted_path, indices)

    print("[SUCCESS] Chaotic Image Encryption & Decryption Completed!")
