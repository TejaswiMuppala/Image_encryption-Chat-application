#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A Telegram client with fingerprint-based image encryption and decryption.
"""

import os
import logging
import asyncio
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError
import cv2
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from skimage.morphology import skeletonize  # For thinning

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Telegram API credentials
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
CHAT_ID = os.getenv('CHAT_ID')  # Default chat ID

# Directory to save received encrypted files
RECEIVED_ENCRYPTED_DIR = "received_encrypted"
os.makedirs(RECEIVED_ENCRYPTED_DIR, exist_ok=True)

# Create the Telegram client
client = TelegramClient('fingerprint_image_crypto_session', API_ID, API_HASH)

# -------------------------------------------
# ENCRYPTION FUNCTIONS (from script 1)
# -------------------------------------------
def generate_aes_key():
    return os.urandom(32)  # 256-bit AES key

def generate_rsa_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key

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

def reverse_logistic_map_biometric(scrambled_image, permutation, original_shape):
    flattened_scrambled = scrambled_image.flatten()
    original_flattened = np.zeros_like(flattened_scrambled)
    inverse_permutation = np.argsort(permutation)
    original_flattened = flattened_scrambled[inverse_permutation]  # Corrected line
    original_image = original_flattened.reshape(original_shape)
    return original_image

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

def aes_decrypt_image(encrypted_image, key, iv, shape):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_padded = decryptor.update(encrypted_image) + decryptor.finalize()
    decrypted_image = np.frombuffer(decrypted_padded, dtype=np.uint8)[:shape[0] * shape[1] * shape[2]].reshape(shape)
    return decrypted_image

# -------------------------------------------
# TELEGRAM CLIENT FUNCTIONS
# -------------------------------------------
async def send_encrypted_image(chat_id, image_path, fingerprint_path):
    """Encrypts an image and sends it along with the encrypted AES key."""
    try:
        print("--- ENCRYPTION PROCESS ---")
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
        biometric_key = biometric_key_raw % 1.0  # Normalize
        if biometric_key == 0.0:
            biometric_key = 0.1

        print(f"[INFO] Number of minutiae: {num_minutiae}")
        print(f"[INFO] Biometric key: {biometric_key}")

        print("[INFO] Generating AES Key...")
        aes_key = generate_aes_key()

        print("[INFO] Generating RSA Key Pair...")
        private_key, public_key = generate_rsa_key_pair()

        print("[INFO] Encrypting AES Key using RSA...")
        encrypted_aes_key = encrypt_aes_key(aes_key, public_key)

        print("[INFO] Encrypting Image with Logistic Map (Biometric Influence)...")
        scrambled_image, permutation = logistic_map_scramble_biometric(image.copy(), biometric_key, num_minutiae)

        print("[INFO] Encrypting Image with AES...")
        iv, encrypted_image = aes_encrypt_image(scrambled_image, aes_key)

        # Save encrypted data to files
        encrypted_image_filename = "encrypted_image.bin"
        encrypted_aes_key_filename = "encrypted_aes_key.bin"
        biometric_key_filename = "biometric_key.txt"
        permutation_filename = "scramble_permutation.npy"
        original_shape_filename = "original_shape.txt"

        np.save(permutation_filename, permutation)
        with open(encrypted_image_filename, "wb") as f:
            f.write(encrypted_image)
        with open(encrypted_aes_key_filename, "wb") as f:
            f.write(encrypted_aes_key)
        with open(biometric_key_filename, "w") as f:
            f.write(str(biometric_key))
        with open(original_shape_filename, "w") as f:
            f.write(f"{image.shape[0]},{image.shape[1]},{image.shape[2]}")

        # Send the files via Telegram
        await client.send_file(chat_id, encrypted_image_filename, caption="Encrypted Image")
        await client.send_file(chat_id, encrypted_aes_key_filename, caption="Encrypted AES Key")
        await client.send_file(chat_id, biometric_key_filename, caption="Biometric Key (for verification)")
        await client.send_file(chat_id, permutation_filename, caption="Scramble Permutation")
        await client.send_file(chat_id, original_shape_filename, caption="Original Image Shape")

        print("[SUCCESS] Image Encrypted and Sent!")

        # Clean up temporary files
        os.remove(encrypted_image_filename)
        os.remove(encrypted_aes_key_filename)
        os.remove(biometric_key_filename)
        os.remove(permutation_filename)
        os.remove(original_shape_filename)

    except Exception as e:
        logger.error(f"Error sending encrypted image: {e}")

@client.on(events.NewMessage(incoming=True))
async def handle_new_message(event):
    """Handles incoming messages to check for encrypted image and key."""
    if event.from_id is None:
        return

    sender = await event.get_sender()
    sender_id = sender.id

    encrypted_image_path = None
    encrypted_aes_key_path = None
    biometric_key_path = None
    permutation_path = None
    original_shape_path = None

    for media in event.media:
        if hasattr(media, 'document'):
            filename = media.document.attributes[0].file_name if media.document.attributes else None
            if filename:
                if filename == "encrypted_image.bin":
                    encrypted_image_path = os.path.join(RECEIVED_ENCRYPTED_DIR, f"{sender_id}_encrypted_image.bin")
                    await client.download_media(media, encrypted_image_path)
                elif filename == "encrypted_aes_key.bin":
                    encrypted_aes_key_path = os.path.join(RECEIVED_ENCRYPTED_DIR, f"{sender_id}_encrypted_aes_key.bin")
                    await client.download_media(media, encrypted_aes_key_path)
                elif filename == "biometric_key.txt":
                    biometric_key_path = os.path.join(RECEIVED_ENCRYPTED_DIR, f"{sender_id}_biometric_key.txt")
                    await client.download_media(media, biometric_key_path)
                elif filename == "scramble_permutation.npy":
                    permutation_path = os.path.join(RECEIVED_ENCRYPTED_DIR, f"{sender_id}_scramble_permutation.npy")
                    await client.download_media(media, permutation_path)
                elif filename == "original_shape.txt":
                    original_shape_path = os.path.join(RECEIVED_ENCRYPTED_DIR, f"{sender_id}_original_shape.txt")
                    await client.download_media(media, original_shape_path)

    if encrypted_image_path and encrypted_aes_key_path and biometric_key_path and permutation_path and original_shape_path:
        print("\n--- DECRYPTION PROCESS (Incoming Message) ---")
        recipient_fingerprint_path = input("Enter path to your fingerprint image for decryption: ").strip()

        try:
            with open(biometric_key_path, "r") as f:
                biometric_key_encrypted = float(f.read())
            permutation_loaded = np.load(permutation_path)
            with open(original_shape_path, "r") as f:
                shape_str = f.read().split(',')
                original_shape = (int(shape_str[0]), int(shape_str[1]), int(shape_str[2]))

            print("[INFO] Extracting your fingerprint features...")
            minutiae_decrypt = extract_fingerprint_features_cn(recipient_fingerprint_path)

            if minutiae_decrypt is None or not minutiae_decrypt:
                print("[ERROR] Could not extract your fingerprint features or no features found.")
                return

            num_minutiae_decrypt = len(minutiae_decrypt)
            coordinates_decrypt = [(point['x'], point['y']) for point in minutiae_decrypt]
            biometric_key_raw_decrypt = np.mean([sum(coord) for coord in coordinates_decrypt]) / (num_minutiae_decrypt + 1e-9) if coordinates_decrypt else 0.7
            biometric_key_decrypt = biometric_key_raw_decrypt % 1.0
            if biometric_key_decrypt == 0.0:
                biometric_key_decrypt = 0.1

            print(f"[INFO] Your Biometric key: {biometric_key_decrypt}")
            print(f"[INFO] Sender's Biometric key: {biometric_key_encrypted}")

            # Check if fingerprint features match (using a threshold)
            if abs(biometric_key_decrypt - biometric_key_encrypted) > 0.01:
                print("[ERROR] Fingerprints do not match. Decryption failed.")
                await event.reply("Decryption failed: Fingerprint verification failed.")
                return

            print("[INFO] Generating RSA Key Pair for decryption...")
            # For decryption, we need the private key. In a real scenario, the recipient would have their own key pair.
            # For this example, we'll generate a new pair for demonstration, but this is NOT secure for real use.
            private_key_decrypt, public_key_decrypt = generate_rsa_key_pair()

            with open(encrypted_aes_key_path, "rb") as f:
                encrypted_aes_key_loaded = f.read()

            print("[INFO] Decrypting AES Key using RSA...")
            try:
                decrypted_aes_key = decrypt_aes_key(encrypted_aes_key_loaded, private_key_decrypt)
            except Exception as e:
                print(f"[ERROR] RSA decryption failed: {e}")
                await event.reply("Decryption failed: Could not decrypt the AES key.")
                return

            with open(encrypted_image_path, "rb") as f:
                encrypted_image_loaded = f.read()

            print("[INFO] Decrypting Image with AES...")
            # We need the IV used during encryption. For simplicity, we'll assume it was prepended or is known.
            # In a real implementation, the IV should be handled more securely (e.g., prepended to the ciphertext).
            iv = encrypted_image_loaded[:16]
            encrypted_content = encrypted_image_loaded[16:]
            decrypted_scrambled_image = aes_decrypt_image(encrypted_content, decrypted_aes_key, iv, original_shape)

            print("[INFO] Reversing Chaotic Scrambling...")
            decrypted_image = reverse_logistic_map_biometric(decrypted_scrambled_image, permutation_loaded, original_shape)

            # Save decrypted image
            decrypted_filename = f"decrypted_image_{sender_id}.jpg"
            cv2.imwrite(decrypted_filename, decrypted_image)
            await client.send_file(event.chat_id, decrypted_filename, caption="Decrypted Image")
            print(f"[SUCCESS] Image Decrypted and Saved as {decrypted_filename}!")
            await event.reply("Image decrypted successfully!")

            # Clean up temporary files
            os.remove(encrypted_image_path)
            os.remove(encrypted_aes_key_path)
            os.remove(biometric_key_path)
            os.remove(permutation_path)
            os.remove(original_shape_path)
            os.remove(decrypted_filename)

        except FileNotFoundError:
            print("[ERROR] Could not find one or more encrypted files.")
            await event.reply("Decryption failed: Missing encrypted files.")
        except Exception as e:
            print(f"[ERROR] Decryption process failed: {e}")
            await event.reply(f"Decryption failed: {e}")

async def main():
    """Main function to run the client and handle sending."""
    await client.connect()

    if not await client.is_user_authorized():
        phone = input("Enter your phone number (with country code): ")
        await client.send_code_request(phone)
        await client.sign_in(phone, input('Enter the code you received: '))

    print("Telegram client started. You can now send encrypted images.")
    print("To send an encrypted image, type 'send_encrypted' followed by the recipient's chat ID and the image path.")

    @client.on(events.NewMessage(pattern='(?i)send_encrypted (\\d+) (.+)'))
    async def send_command_handler(event):
        chat_id = int(event.pattern_match.group(1))
        image_path = event.pattern_match.group(2)
        fingerprint_path = input("Enter path to your fingerprint image for encryption: ").strip()
        await send_encrypted_image(chat_id, image_path, fingerprint_path)

    await client.run_until_disconnected()

if __name__ == '__main__':
    asyncio.run(main())