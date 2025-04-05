import cv2
import numpy as np
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from skimage.morphology import skeletonize # For thinning

# -------------------------------------------
# 1. GENERATE AES KEY
# -------------------------------------------
def generate_aes_key():
    return os.urandom(32)  # 256-bit AES key

# -------------------------------------------
# 2. RSA KEY PAIR GENERATION
# -------------------------------------------
def generate_rsa_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key

# -------------------------------------------
# 3. ENCRYPT AES KEY USING RSA
# -------------------------------------------
def encrypt_aes_key(aes_key, public_key):
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted_key

# -------------------------------------------
# 4. DECRYPT AES KEY USING RSA
# -------------------------------------------
def decrypt_aes_key(encrypted_key, private_key):
    decrypted_key = private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return decrypted_key

# -------------------------------------------
# 5. FINGERPRINT FEATURE EXTRACTION (Crossing Number Method)
# -------------------------------------------
def detect_minutiae_cn(thinned_image):
    minutiae = []
    height, width = thinned_image.shape

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if thinned_image[i, j] == 255:  # Ridge pixel
                neighborhood = [
                    thinned_image[i - 1, j - 1] // 255,
                    thinned_image[i - 1, j] // 255,
                    thinned_image[i - 1, j + 1] // 255,
                    thinned_image[i, j + 1] // 255,
                    thinned_image[i + 1, j + 1] // 255,
                    thinned_image[i + 1, j] // 255,
                    thinned_image[i + 1, j - 1] // 255,
                    thinned_image[i, j - 1] // 255
                ]

                cn = 0
                for k in range(8):
                    cn += abs(neighborhood[k] - neighborhood[(k + 1) % 8])

                cn //= 2

                if cn == 1:
                    minutiae.append({'type': 'ending', 'x': j, 'y': i})
                elif cn == 3:
                    minutiae.append({'type': 'bifurcation', 'x': j, 'y': i})

    return minutiae

def extract_fingerprint_features_cn(fingerprint_path):
    try:
        img = cv2.imread(fingerprint_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[ERROR] Could not load fingerprint image: {fingerprint_path}")
            return None

        # 1. Preprocessing
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        equalized = cv2.equalizeHist(blurred)

        # 2. Segmentation and Binarization
        _, binary_img = cv2.threshold(equalized, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Ridge Thinning (using skeletonize from skimage)
        thinned = skeletonize(binary_img // 255).astype(np.uint8) * 255

        # 4. Minutiae Detection using Crossing Number
        minutiae = detect_minutiae_cn(thinned)

        return minutiae

    except Exception as e:
        print(f"[ERROR] Fingerprint feature extraction failed: {e}")
        return None

# -------------------------------------------
# 6. CHAOTIC MAP FUNCTION (LOGISTIC MAP) WITH BIOMETRIC INFLUENCE
# -------------------------------------------
def logistic_map_scramble_biometric(image, biometric_key, num_minutiae):
    height, width, channels = image.shape
    total_pixels = height * width * channels  # Total number of pixels

    # Generate chaotic sequence with length equal to the number of minutiae
    chaotic_sequence = np.zeros(num_minutiae)
    x = biometric_key
    r = 3.99

    for i in range(num_minutiae):
        x = r * x * (1 - x)
        chaotic_sequence[i] = x

    # Create a permutation of pixel indices based on the chaotic sequence
    rng = np.random.RandomState(int(np.sum(chaotic_sequence) * 100000) % (2**32 - 1) if num_minutiae > 0 else 0)
    permutation = rng.permutation(total_pixels)

    # Scramble the image
    flattened_image = image.flatten()
    scrambled_flattened = flattened_image[permutation]
    scrambled_image = scrambled_flattened.reshape(image.shape)

    return scrambled_image, permutation

# -------------------------------------------
# 7. REVERSE CHAOTIC SCRAMBLING
# -------------------------------------------
def reverse_logistic_map_biometric(scrambled_image, permutation, original_shape):
    flattened_scrambled = scrambled_image.flatten()
    original_flattened = np.zeros_like(flattened_scrambled)
    inverse_permutation = np.argsort(permutation)
    original_flattened = flattened_scrambled[inverse_permutation] # Corrected line
    original_image = original_flattened.reshape(original_shape)
    return original_image

# -------------------------------------------
# 8. AES IMAGE ENCRYPTION
# -------------------------------------------
def aes_encrypt_image(image, key):
    iv = os.urandom(16)  # Initialization vector
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Padding image
    height, width, channels = image.shape
    padded_size = (height * width * channels + 16) // 16 * 16
    padded_image = np.pad(image.flatten(), (0, padded_size - height * width * channels), mode='constant')

    encrypted_image = encryptor.update(padded_image.tobytes()) + encryptor.finalize()
    return iv, encrypted_image

# -------------------------------------------
# 9. AES IMAGE DECRYPTION
# -------------------------------------------
def aes_decrypt_image(encrypted_image, key, iv, shape):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded = decryptor.update(encrypted_image) + decryptor.finalize()
    decrypted_image = np.frombuffer(decrypted_padded, dtype=np.uint8)[:shape[0] * shape[1] * shape[2]].reshape(shape)
    return decrypted_image

# -------------------------------------------
# 10. MAIN FUNCTION TO RUN HYBRID ENCRYPTION WITH BIOMETRICS AND LOGISTIC MAP
# -------------------------------------------
def main():
    # --- ENCRYPTION ---
    print("--- ENCRYPTION PROCESS ---")
    image_path = input("Enter path of the image to encrypt: ").strip()
    fingerprint_path = input("Enter path to the user's fingerprint image: ").strip()
    image = cv2.imread(image_path)

    if image is None:
        print("[ERROR] Could not load image!")
        return

    print("[INFO] Extracting fingerprint features...")
    minutiae = extract_fingerprint_features_cn(fingerprint_path)

    if minutiae is None or not minutiae:
        print("[ERROR] Could not extract fingerprint features or no features found.")
        return

    num_minutiae = len(minutiae)
    coordinates = [(point['x'], point['y']) for point in minutiae]
    biometric_key_raw = np.mean([sum(coord) for coord in coordinates]) / (num_minutiae + 1e-9) if coordinates else 0.7
    # Normalize the biometric key to be within (0, 1)
    biometric_key = biometric_key_raw % 1.0 # Taking the fractional part
    if biometric_key == 0.0:
        biometric_key = 0.1 # Avoid key being exactly 0

    print(f"[INFO] Number of minutiae: {num_minutiae}")
    print(f"[INFO] Raw Biometric key: {biometric_key_raw}")
    print(f"[INFO] Normalized Biometric key: {biometric_key}")

    print("[INFO] Generating AES Key...")
    aes_key = generate_aes_key()

    print("[INFO] Generating RSA Key Pair...")
    private_key, public_key = generate_rsa_key_pair()

    print("[INFO] Encrypting AES Key using RSA...")
    encrypted_aes_key = encrypt_aes_key(aes_key, public_key)

    # Apply chaotic scrambling with Logistic Map, influenced by biometrics
    print("[INFO] Encrypting Image with Logistic Map (Biometric Influence)...")
    scrambled_image, permutation = logistic_map_scramble_biometric(image.copy(), biometric_key, num_minutiae)

    # Apply AES Encryption
    print("[INFO] Encrypting Image with AES...")
    iv, encrypted_image = aes_encrypt_image(scrambled_image, aes_key)

    # Save encrypted data
    cv2.imwrite("hybrid_encrypted.jpg", scrambled_image)  # Save scrambled image (optional for visualization)
    with open("encrypted_aes_key.bin", "wb") as f:
        f.write(encrypted_aes_key)
    with open("encrypted_image.bin", "wb") as f:
        f.write(encrypted_image)
    with open("scramble_permutation.bin", "wb") as f: # Save permutation for decryption
        np.save(f, permutation)
    with open("minutiae_count.txt", "w") as f: # Save minutiae count
        f.write(str(num_minutiae))

    print("[SUCCESS] Image Encrypted and Saved!")

    # --- DECRYPTION ---
    print("\n--- DECRYPTION PROCESS ---")
    encrypted_image_path = "encrypted_image.bin"
    encrypted_key_path = "encrypted_aes_key.bin"
    permutation_path = "scramble_permutation.bin"
    minutiae_count_path = "minutiae_count.txt"
    fingerprint_path_decrypt = input("Enter path to your fingerprint image for decryption: ").strip()

    try:
        with open(encrypted_key_path, "rb") as f:
            encrypted_aes_key_loaded = f.read()
        with open(encrypted_image_path, "rb") as f:
            encrypted_image_loaded = f.read()
        permutation_loaded = np.load(permutation_path)
        with open(minutiae_count_path, "r") as f:
            num_minutiae_encrypted = int(f.read())

        print("[INFO] Extracting fingerprint features for decryption...")
        minutiae_decrypt = extract_fingerprint_features_cn(fingerprint_path_decrypt)

        if minutiae_decrypt is None or not minutiae_decrypt:
            print("[ERROR] Could not extract fingerprint features for decryption or no features found.")
            return

        num_minutiae_decrypt = len(minutiae_decrypt)
        coordinates_decrypt = [(point['x'], point['y']) for point in minutiae_decrypt]
        biometric_key_raw_decrypt = np.mean([sum(coord) for coord in coordinates_decrypt]) / (num_minutiae_decrypt + 1e-9) if coordinates_decrypt else 0.7
        biometric_key_decrypt = biometric_key_raw_decrypt % 1.0
        if biometric_key_decrypt == 0.0:
            biometric_key_decrypt = 0.1

        print(f"[INFO] Number of minutiae (decryption): {num_minutiae_decrypt}")
        print(f"[INFO] Raw Biometric key (decryption): {biometric_key_raw_decrypt}")
        print(f"[INFO] Normalized Biometric key (decryption): {biometric_key_decrypt}")
        print(f"[INFO] Number of minutiae (encrypted): {num_minutiae_encrypted}")

        # **CRITICAL: Check if the number of minutiae matches.**
        if num_minutiae_encrypted != num_minutiae_decrypt:
            print("[ERROR] Fingerprints do not match. Decryption failed.")
            return

        print("[INFO] Decrypting AES Key using RSA...")
        decrypted_aes_key = decrypt_aes_key(encrypted_aes_key_loaded, private_key)

        print("[INFO] Decrypting Image with AES...")
        original_shape = cv2.imread(image_path).shape # Get original shape for decryption
        decrypted_scrambled_image = aes_decrypt_image(encrypted_image_loaded, decrypted_aes_key, iv, original_shape)

        print("[INFO] Reversing Chaotic Scrambling...")
        decrypted_image = reverse_logistic_map_biometric(decrypted_scrambled_image, permutation_loaded, original_shape)

        # Save decrypted image
        cv2.imwrite("hybrid_decrypted.jpg", decrypted_image)
        print("[SUCCESS] Image Decrypted and Saved!")

    except FileNotFoundError:
        print("[ERROR] Encrypted files not found!")
    except Exception as e:
        print(f"[ERROR] Decryption failed: {e}")

# -------------------------------------------
# RUN THE PROGRAM
# -------------------------------------------
if __name__ == "__main__":
    main()