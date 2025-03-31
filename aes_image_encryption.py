import os
import numpy as np
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

# Load the image
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')  # Convert to RGB
    img_array = np.array(img)
    return img, img_array

# Convert image to byte data
def image_to_bytes(img_array):
    return img_array.tobytes()

# Convert byte data back to image
def bytes_to_image(byte_data, shape):
    return Image.fromarray(np.frombuffer(byte_data, dtype=np.uint8).reshape(shape))

# AES Encryption
def encrypt_image(img_bytes, key, iv):
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(img_bytes) + padder.finalize()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    
    return encrypted_data

# AES Decryption
def decrypt_image(encrypted_data, key, iv, shape):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded_data = decryptor.update(encrypted_data) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    decrypted_data = unpadder.update(decrypted_padded_data) + unpadder.finalize()

    return bytes_to_image(decrypted_data, shape)

# Main function
def main():
    image_path = r"C:\Users\Tejaswi\OneDrive\Documents\SEM-VI\computer security\project\9d15cda706c2158079156ce5a88bfc9a.jpg" # Change to your image path
    img, img_array = load_image(image_path)

    key = os.urandom(32)  # Generate a 256-bit AES key
    iv = os.urandom(16)   # Generate a 128-bit IV
    

    print("[INFO] Encrypting Image...")
    img_bytes = image_to_bytes(img_array)
    encrypted_data = encrypt_image(img_bytes, key, iv)

    print("[INFO] Decrypting Image...")
    decrypted_img = decrypt_image(encrypted_data, key, iv, img_array.shape)

    # Save and Show the Decrypted Image
    decrypted_img.save("decrypted_image.jpg")
    decrypted_img.show()

if __name__ == "__main__":
    main()
