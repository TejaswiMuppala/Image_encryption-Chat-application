import os
import numpy as np
from PIL import Image
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives import serialization

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

# RSA Key Generation
def generate_rsa_keys():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()

    # Save keys to files
    with open("rsa_private_key.pem", "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    with open("rsa_public_key.pem", "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))

    return private_key, public_key

# RSA Encryption (Encrypt AES Key)
def encrypt_aes_key(aes_key, public_key):
    encrypted_key = public_key.encrypt(
        aes_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=algorithms.SHA256()),
            algorithm=algorithms.SHA256(),
            label=None
        )
    )
    return encrypted_key

# RSA Decryption (Decrypt AES Key)
def decrypt_aes_key(encrypted_aes_key, private_key):
    decrypted_key = private_key.decrypt(
        encrypted_aes_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=algorithms.SHA256()),
            algorithm=algorithms.SHA256(),
            label=None
        )
    )
    return decrypted_key

# Main function
def main():
    image_path = r"C:\Users\Tejaswi\OneDrive\Documents\SEM-VI\computer security\project\9d15cda706c2158079156ce5a88bfc9a.jpg"  # Change to your image path
    img, img_array = load_image(image_path)

    # Generate AES Key & IV
    aes_key = os.urandom(32)  # 256-bit AES key
    iv = os.urandom(16)  # 128-bit IV

    # Generate RSA Key Pair
    print("[INFO] Generating RSA Key Pair...")
    private_key, public_key = generate_rsa_keys()

    # Encrypt AES Key using RSA
    print("[INFO] Encrypting AES Key using RSA...")
    encrypted_aes_key = encrypt_aes_key(aes_key, public_key)

    # Encrypt Image using AES
    print("[INFO] Encrypting Image...")
    img_bytes = image_to_bytes(img_array)
    encrypted_data = encrypt_image(img_bytes, aes_key, iv)

    # Save encrypted data to a file
    with open("encrypted_image.bin", "wb") as f:
        f.write(encrypted_data)

    with open("encrypted_aes_key.bin", "wb") as f:
        f.write(encrypted_aes_key)

    print("[INFO] Encrypted image and AES key saved!")

    # ==================== Decryption ==================== #

    # Load Encrypted AES Key
    with open("encrypted_aes_key.bin", "rb") as f:
        encrypted_aes_key = f.read()

    # Decrypt AES Key using RSA Private Key
    print("[INFO] Decrypting AES Key using RSA...")
    decrypted_aes_key = decrypt_aes_key(encrypted_aes_key, private_key)

    # Decrypt Image
    print("[INFO] Decrypting Image...")
    decrypted_img = decrypt_image(encrypted_data, decrypted_aes_key, iv, img_array.shape)

    # Save and Show the Decrypted Image
    decrypted_img.save("decrypted_image.jpg")
    decrypted_img.show()

if __name__ == "__main__":
    main()
