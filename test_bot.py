import os
import logging
import asyncio
import numpy as np
import cv2
from dotenv import load_dotenv
from telethon import TelegramClient, events, Button
from telethon.tl.types import InputFile
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from skimage.morphology import skeletonize

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Get credentials from environment variables
BOT_TOKEN = os.getenv('BOT_TOKEN')
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')

class SecureImageBot:
    def __init__(self):
        self.client = TelegramClient('bot_session', API_ID, API_HASH)
        self.private_key = None
        self.public_key = None
        self.user_states = {}  # Track user registration states
        self.pending_encryptions = {}  # Track pending image encryptions
        self.user_fingerprints = {}  # Store fingerprint data securely
        
        # Create required directories
        self.dirs = {
            'temp': 'temp_files',
            'fingerprints': 'private/fingerprints',
            'encrypted': 'private/encrypted'
        }
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    async def start(self):
        """Start the bot."""
        # Generate RSA keys
        self.private_key, self.public_key = self._generate_rsa_keys()
        
        # Set up event handlers
        self.client.add_event_handler(self._handle_private_message, 
                                    events.NewMessage(pattern='/start|/send', func=lambda e: e.is_private))
        self.client.add_event_handler(self._handle_register, 
                                    events.NewMessage(pattern='/register', func=lambda e: e.is_private))
        self.client.add_event_handler(self._handle_group_message, 
                                    events.NewMessage(func=lambda e: not e.is_private))
        self.client.add_event_handler(self._handle_decrypt_request, 
                                    events.NewMessage(pattern='/decrypt', func=lambda e: e.is_private))
        self.client.add_event_handler(self._handle_media, 
                                    events.NewMessage(func=lambda e: e.is_private and (e.photo or e.document)))
        self.client.add_event_handler(self._handle_text_message,
                                    events.NewMessage(func=lambda e: e.is_private and e.text and not e.text.startswith('/')))
        
        # Start the client
        await self.client.start(bot_token=BOT_TOKEN)
        logging.info("Bot started successfully!")
        await self.client.run_until_disconnected()

    def _generate_rsa_keys(self):
        """Generate RSA key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        return private_key, private_key.public_key()

    async def _handle_private_message(self, event):
        """Handle private messages."""
        if event.raw_text == '/start':
            await event.respond(
                "🔒 Welcome to Secure Image Bot!\n\n"
                "To use secure image sharing:\n"
                "1. First register your fingerprint privately using /register\n"
                "2. Use /send to send encrypted images to friends\n"
                "3. Use /decrypt to decrypt received images\n\n"
                "Commands:\n"
                "/register - Register your fingerprint\n"
                "/send - Send encrypted image to a friend\n"
                "/decrypt - Decrypt received images"
            )
        elif event.raw_text == '/send':
            if event.sender_id not in self.user_fingerprints:
                await event.respond(
                    "⚠️ You need to register first!\n"
                    "Use /register to set up secure messaging."
                )
                return
            
            self.user_states[event.sender_id] = {
                'state': 'awaiting_recipient',
                'type': 'send'
            }
            await event.respond(
                "Forward any message from your friend or send their username\n"
                "This is who will receive your encrypted image."
            )

    async def _handle_register(self, event):
        """Handle fingerprint registration."""
        user_id = event.sender_id
        
        # Initialize registration state
        self.user_states[user_id] = {'state': 'awaiting_fingerprint'}
        
        await event.respond(
            "Please send a clear image of your fingerprint.\n"
            "This will be used to secure your messages.\n"
            "⚠️ Send this in our private chat only!"
        )

    async def _process_fingerprint(self, event):
        """Process fingerprint registration."""
        user_id = event.sender_id
        
        try:
            # Download fingerprint image
            temp_path = os.path.join(self.dirs['temp'], f'fp_{user_id}.jpg')
            await event.download_media(temp_path)
            
            # Process fingerprint
            features = self._extract_fingerprint_features(temp_path)
            if not features:
                await event.respond("❌ Could not process fingerprint. Please send a clearer image.")
                return False
            
            # Generate and store biometric key
            coordinates = [(p['x'], p['y']) for p in features]
            bio_key = np.mean([sum(coord) for coord in coordinates]) / (len(features) + 1e-9)
            bio_key = bio_key % 1.0 or 0.1
            
            # Store user's fingerprint data securely
            self.user_fingerprints[user_id] = {
                'bio_key': bio_key,
                'minutiae_count': len(features)
            }
            
            await event.respond(
                "✅ Fingerprint registered successfully!\n\n"
                "You can now:\n"
                "1. Use /send to send encrypted images to friends\n"
                "2. Use /decrypt to decrypt received images\n\n"
                "Try sending an image by using the /send command!"
            )
            return True
            
        except Exception as e:
            logging.error(f"Fingerprint processing error: {e}")
            await event.respond("❌ Error processing fingerprint. Please try again.")
            return False
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _extract_fingerprint_features(self, image_path):
        """Extract fingerprint features."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None

            # Process image
            img = cv2.resize(img, (500, 500))
            img = cv2.equalizeHist(img)
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            skeleton = skeletonize(binary > 0).astype(np.uint8) * 255
            
            return self._detect_minutiae(skeleton)
            
        except Exception as e:
            logging.error(f"Feature extraction error: {e}")
            return None

    def _detect_minutiae(self, skeleton):
        """Detect minutiae points in fingerprint."""
        minutiae = []
        height, width = skeleton.shape

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if skeleton[i, j] == 255:
                    # Get 8-connected neighborhood
                    neighbors = np.array([
                        skeleton[i-1, j-1],
                        skeleton[i-1, j],
                        skeleton[i-1, j+1],
                        skeleton[i, j+1],
                        skeleton[i+1, j+1],
                        skeleton[i+1, j],
                        skeleton[i+1, j-1],
                        skeleton[i, j-1]
                    ], dtype=np.float32) / 255.0

                    # Calculate crossing number
                    cn = 0
                    for k in range(8):
                        cn += abs(int(neighbors[k]) - int(neighbors[(k+1) % 8]))
                    cn //= 2

                    if cn == 1:  # Ridge ending
                        minutiae.append({'type': 'ending', 'x': j, 'y': i})
                    elif cn == 3:  # Bifurcation
                        minutiae.append({'type': 'bifurcation', 'x': j, 'y': i})

        return minutiae

    async def _handle_group_message(self, event):
        """Handle group messages with images."""
        # Ignore group messages since we're focusing on private messaging only
        return

    async def _handle_decrypt_request(self, event):
        """Handle decryption requests."""
        user_id = event.sender_id

        # Check if user is registered
        if user_id not in self.user_fingerprints:
            await event.respond(
                "⚠️ You need to register first!\n"
                "Use /register and follow the instructions."
            )
            return

        # Initialize decryption state
        self.user_states[user_id] = {
            'state': 'awaiting_files',
            'files': []
        }

        await event.respond(
            "Please forward all encrypted files here.\n"
            "I need all 4 files to decrypt the image."
        )

    async def _handle_text_message(self, event):
        """Handle text messages for recipient selection."""
        user_id = event.sender_id
        
        if user_id not in self.user_states:
            return
            
        state = self.user_states[user_id]
        
        # Handle recipient selection
        if state.get('state') == 'awaiting_recipient':
            try:
                username = event.text.strip('@')
                try:
                    recipient = await self.client.get_entity(username)
                    state['recipient'] = recipient
                    state['state'] = 'awaiting_image'
                    await event.respond(
                        "✅ Recipient selected!\n"
                        "Now send the image you want to encrypt and share."
                    )
                except ValueError:
                    await event.respond(
                        "❌ Could not find user.\n"
                        "Please make sure to:\n"
                        "1. Send their exact Telegram username (with or without @)\n"
                        "2. Or forward any message from them"
                    )
            except Exception as e:
                logging.error(f"Error in recipient selection: {e}")
                await event.respond("❌ Error finding recipient. Please try again.")

    async def _handle_recipient_selection(self, event):
        """Handle recipient selection for private sending."""
        user_id = event.sender_id
        
        if user_id not in self.user_states or self.user_states[user_id].get('state') != 'awaiting_recipient':
            return
            
        state = self.user_states[user_id]
        
        try:
            # Handle forwarded message
            if event.forward:
                if hasattr(event.forward.from_id, 'user_id'):
                    recipient_id = event.forward.from_id.user_id
                    try:
                        recipient = await self.client.get_entity(recipient_id)
                        state['recipient'] = recipient
                        state['state'] = 'awaiting_image'
                        await event.respond(
                            "✅ Recipient selected!\n"
                            "Now send the image you want to encrypt and share."
                        )
                        return
                    except Exception as e:
                        logging.error(f"Error getting recipient from forward: {e}")
                
            # Handle username
            if event.text:
                username = event.text.strip('@')
                try:
                    recipient = await self.client.get_entity(username)
                    state['recipient'] = recipient
                    state['state'] = 'awaiting_image'
                    await event.respond(
                        "✅ Recipient selected!\n"
                        "Now send the image you want to encrypt and share."
                    )
                    return
                except ValueError as e:
                    logging.error(f"Invalid username: {e}")
                except Exception as e:
                    logging.error(f"Error getting recipient from username: {e}")
            
            # If we get here, neither method worked
            await event.respond(
                "❌ Could not find recipient.\n"
                "Please either:\n"
                "1. Forward any message from them, or\n"
                "2. Send their exact Telegram username (with or without @)\n\n"
                "Example: @username or just username"
            )

        except Exception as e:
            logging.error(f"Error in recipient selection: {e}")
            await event.respond(
                "❌ Error finding recipient.\n"
                "Please try again with a forwarded message or correct username."
            )

    async def _handle_media(self, event):
        """Handle media messages in private chat."""
        user_id = event.sender_id
        
        if user_id not in self.user_states:
            return
            
        state = self.user_states[user_id]
        
        # Handle fingerprint registration
        if state.get('state') == 'awaiting_fingerprint':
            if event.photo:
                await self._process_fingerprint(event)
            else:
                await event.respond("Please send a photo of your fingerprint.")
            return
            
        # Handle image for encryption and sending
        if state.get('state') == 'awaiting_image' and state.get('type') == 'send':
            if not event.photo:
                await event.respond("Please send an image to encrypt.")
                return
                
            try:
                # Download image
                temp_path = os.path.join(self.dirs['temp'], f'img_{event.id}.jpg')
                await event.download_media(temp_path)
                
                # Get sender's biometric data
                bio_data = self.user_fingerprints[user_id]
                
                # Encrypt image
                encrypted_files = await self._encrypt_image(
                    temp_path, 
                    bio_data['bio_key'],
                    bio_data['minutiae_count']
                )
                
                # Send encrypted files to recipient
                recipient = state['recipient']
                await event.respond("🔒 Sending encrypted files to recipient...")
                
                for file_path in encrypted_files:
                    await self.client.send_file(
                        recipient,
                        file_path,
                        force_document=True,
                        caption="🔒 Encrypted image file"
                    )
                
                await event.respond(
                    "✅ Encrypted image sent!\n"
                    "Your friend needs to:\n"
                    "1. Register their fingerprint with me\n"
                    "2. Forward the files to me\n"
                    "3. Use /decrypt to view the image"
                )
                
                # Clean up
                os.remove(temp_path)
                for file in encrypted_files:
                    if os.path.exists(file):
                        os.remove(file)
                        
                # Clear state
                del self.user_states[user_id]
                
            except Exception as e:
                await event.respond("❌ Failed to encrypt and send image. Please try again.")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            return
            
        # Handle encrypted files for decryption
        if state.get('state') == 'awaiting_files':
            if not event.document:
                await event.respond("Please forward the encrypted files.")
                return
                
            # Download file
            temp_path = os.path.join(self.dirs['temp'], f'dec_{event.id}_{event.document.attributes[0].file_name}')
            await event.download_media(temp_path)
            state['files'].append(temp_path)
            
            # Check if we have all files
            if len(state['files']) < 4:
                await event.respond(f"✅ File {len(state['files'])}/4 received. Send remaining files.")
            else:
                # Process decryption
                await self._process_decryption(event, state['files'])
                # Clear state
                del self.user_states[user_id]

    async def _process_decryption(self, event, files):
        """Process decryption of received files."""
        try:
            # Map files to their types
            file_map = {}
            for f in files:
                if f.endswith('_image.bin'):
                    file_map['image'] = f
                elif f.endswith('_key.bin'):
                    file_map['key'] = f
                elif f.endswith('_meta.txt'):
                    file_map['meta'] = f
                elif f.endswith('_perm.npy'):
                    file_map['perm'] = f

            if len(file_map) != 4:
                await event.respond("❌ Missing some required files. Please send all 4 encrypted files.")
                return

            # Read metadata
            with open(file_map['meta'], 'r') as f:
                stored_minutiae = int(f.readline().strip())
                iv_hex = f.readline().strip()
                height = int(f.readline().strip())
                width = int(f.readline().strip())
                channels = int(f.readline().strip())

            # Verify fingerprint match
            user_data = self.user_fingerprints[event.sender_id]
            if stored_minutiae != user_data['minutiae_count']:
                await event.respond("❌ Decryption failed: Fingerprint verification failed.")
                return

            # Load encrypted data
            with open(file_map['image'], 'rb') as f:
                encrypted_image = f.read()
            with open(file_map['key'], 'rb') as f:
                encrypted_key = f.read()
            permutation = np.load(file_map['perm'])
            iv = bytes.fromhex(iv_hex)

            # Decrypt AES key
            aes_key = self._decrypt_aes_key(encrypted_key)

            # Decrypt image
            shape = (height, width, channels)
            decrypted_scrambled = self._aes_decrypt(encrypted_image, aes_key, iv, shape)
            
            # Unscramble
            decrypted = self._unscramble_image(decrypted_scrambled, permutation, shape)

            # Save and send decrypted image
            output_path = os.path.join(self.dirs['temp'], f'decrypted_{event.id}.jpg')
            cv2.imwrite(output_path, decrypted)
            
            await self.client.send_file(
                event.chat_id,
                output_path,
                caption="🔓 Here's your decrypted image!"
            )

        except Exception as e:
            logging.error(f"Decryption error: {e}")
            await event.respond("❌ Decryption failed. Please try again.")
            
        finally:
            # Clean up files
            try:
                for f in files:
                    if os.path.exists(f):
                        os.remove(f)
                if os.path.exists(output_path):
                    os.remove(output_path)
            except:
                pass

    def _decrypt_aes_key(self, encrypted_key):
        """Decrypt AES key using RSA."""
        return self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    def _aes_decrypt(self, encrypted_data, key, iv, shape):
        """Decrypt data using AES."""
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
        decrypted = np.frombuffer(decrypted_padded, dtype=np.uint8)[:shape[0] * shape[1] * shape[2]]
        return decrypted.reshape(shape)

    def _unscramble_image(self, scrambled_image, permutation, shape):
        """Unscramble image using stored permutation."""
        flattened = scrambled_image.flatten()
        inverse_perm = np.argsort(permutation)
        original = flattened[inverse_perm]
        return original.reshape(shape)

    async def _encrypt_image(self, image_path, bio_key, minutiae_count):
        """Encrypt image using hybrid encryption with biometric influence."""
        try:
            # Read and process image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")

            # Generate session key
            aes_key = os.urandom(32)  # 256-bit AES key
            
            # Scramble image using biometric data
            scrambled, permutation = self._scramble_image(image, bio_key, minutiae_count)
            
            # Encrypt with AES
            iv, encrypted_data = self._aes_encrypt(scrambled, aes_key)
            
            # Encrypt AES key with RSA
            encrypted_key = self._encrypt_aes_key(aes_key)
            
            # Save encrypted files
            timestamp = int(asyncio.get_event_loop().time())
            base_path = os.path.join(self.dirs['encrypted'], f'secure_{timestamp}')
            
            file_paths = []
            
            # Save encrypted image
            img_path = f"{base_path}_image.bin"
            with open(img_path, 'wb') as f:
                f.write(encrypted_data)
            file_paths.append(img_path)
            
            # Save encrypted key
            key_path = f"{base_path}_key.bin"
            with open(key_path, 'wb') as f:
                f.write(encrypted_key)
            file_paths.append(key_path)
            
            # Save permutation
            perm_path = f"{base_path}_perm.npy"
            np.save(perm_path, permutation)
            file_paths.append(perm_path)
            
            # Save metadata (minutiae count, IV, and image shape)
            meta_path = f"{base_path}_meta.txt"
            with open(meta_path, 'w') as f:
                f.write(f"{minutiae_count}\n{iv.hex()}\n{image.shape[0]}\n{image.shape[1]}\n{image.shape[2]}")
            file_paths.append(meta_path)

            return file_paths

        except Exception as e:
            logging.error(f"Encryption error: {e}")
            raise

    def _scramble_image(self, image, bio_key, minutiae_count):
        """Scramble image using chaotic map with biometric influence."""
        height, width, channels = image.shape
        total_pixels = height * width * channels

        # Generate chaotic sequence
        chaotic_sequence = np.zeros(minutiae_count)
        x = bio_key
        r = 3.99

        for i in range(minutiae_count):
            x = r * x * (1 - x)
            chaotic_sequence[i] = x

        # Create permutation based on chaotic sequence
        seed = int(np.sum(chaotic_sequence) * 1000000) % (2**32)
        rng = np.random.RandomState(seed)
        permutation = rng.permutation(total_pixels)

        # Scramble image
        flattened = image.flatten()
        scrambled = flattened[permutation]
        return scrambled.reshape(image.shape), permutation

    def _aes_encrypt(self, image, key):
        """Encrypt image using AES."""
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        # Pad image
        flat = image.flatten()
        pad_size = (len(flat) + 15) // 16 * 16
        padded = np.pad(flat, (0, pad_size - len(flat)))
        
        encrypted = encryptor.update(padded.tobytes()) + encryptor.finalize()
        return iv, encrypted

    def _encrypt_aes_key(self, aes_key):
        """Encrypt AES key using RSA."""
        return self.public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

if __name__ == '__main__':
    # Run the bot
    bot = SecureImageBot()
    asyncio.run(bot.start())