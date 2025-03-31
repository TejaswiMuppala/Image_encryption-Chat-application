import cv2
import numpy as np
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend

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
# 5. CHAOTIC MAP FUNCTION (LOGISTIC MAP)
# -------------------------------------------
def logistic_map_scramble(image, key):
    height, width, channels = image.shape
    total_pixels = height * width * channels  # Total number of pixels
    
    chaotic_sequence = np.zeros(total_pixels)
    x = key

    # Generate chaotic sequence
    for i in range(total_pixels):
        x = 3.99 * x * (1 - x)
        chaotic_sequence[i] = x

    # Convert sequence to pixel permutation indices
    indices = np.argsort(chaotic_sequence)
    
    # Scramble the image
    flattened_image = image.flatten()
    scrambled_flattened = np.zeros_like(flattened_image)
    scrambled_flattened[indices] = flattened_image
    
    scrambled_image = scrambled_flattened.reshape(image.shape)
    return scrambled_image, indices

# -------------------------------------------
# 6. REVERSE CHAOTIC SCRAMBLING
# -------------------------------------------
def reverse_logistic_map(scrambled_image, indices):
    original_shape = scrambled_image.shape
    flattened_scrambled = scrambled_image.flatten()
    
    # Restore original order
    restored_flattened = np.zeros_like(flattened_scrambled)
    restored_flattened[np.argsort(indices)] = flattened_scrambled

    restored_image = restored_flattened.reshape(original_shape)
    return restored_image

# -------------------------------------------
# 7. AES IMAGE ENCRYPTION
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
# 8. AES IMAGE DECRYPTION
# -------------------------------------------
def aes_decrypt_image(encrypted_image, key, iv, shape):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded = decryptor.update(encrypted_image) + decryptor.finalize()
    decrypted_image = np.frombuffer(decrypted_padded, dtype=np.uint8)[:shape[0] * shape[1] * shape[2]].reshape(shape)
    return decrypted_image

# -------------------------------------------
# 9. MAIN FUNCTION TO RUN HYBRID ENCRYPTION
# -------------------------------------------
def main():
    # Load image
    image_path = input("Enter path of the image to encrypt: ").strip()
    image = cv2.imread(image_path)
    
    if image is None:
        print("[ERROR] Could not load image!")
        return

    print("[INFO] Generating AES Key...")
    aes_key = generate_aes_key()

    print("[INFO] Generating RSA Key Pair...")
    private_key, public_key = generate_rsa_key_pair()

    print("[INFO] Encrypting AES Key using RSA...")
    encrypted_aes_key = encrypt_aes_key(aes_key, public_key)

    # Apply chaotic scrambling
    print("[INFO] Encrypting Image with Chaotic Maps...")
    chaotic_key = 0.7  # Initial chaotic map key
    scrambled_image, scramble_indices = logistic_map_scramble(image, chaotic_key)

    # Apply AES Encryption
    print("[INFO] Encrypting Image with AES...")
    iv, encrypted_image = aes_encrypt_image(scrambled_image, aes_key)

    # Save encrypted data
    cv2.imwrite("hybrid_encrypted.jpg", scrambled_image)  # Save scrambled image
    with open("encrypted_aes_key.bin", "wb") as f:
        f.write(encrypted_aes_key)
    with open("encrypted_image.bin", "wb") as f:
        f.write(encrypted_image)

    print("[SUCCESS] Image Encrypted and Saved!")

    # DECRYPTION PROCESS
    print("[INFO] Decrypting AES Key using RSA...")
    decrypted_aes_key = decrypt_aes_key(encrypted_aes_key, private_key)

    print("[INFO] Decrypting Image with AES...")
    decrypted_scrambled_image = aes_decrypt_image(encrypted_image, decrypted_aes_key, iv, scrambled_image.shape)

    print("[INFO] Reversing Chaotic Scrambling...")
    decrypted_image = reverse_logistic_map(decrypted_scrambled_image, scramble_indices)

    # Save decrypted image
    cv2.imwrite("hybrid_decrypted.jpg", decrypted_image)
    print("[SUCCESS] Image Decrypted and Saved!")

# -------------------------------------------
# RUN THE PROGRAM
# -------------------------------------------
if __name__ == "__main__":
    main()
