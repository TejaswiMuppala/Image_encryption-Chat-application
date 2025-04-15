import os
import logging
import asyncio
import numpy as np
import cv2
from telethon import TelegramClient, events, Button
from telethon.tl.types import InputFile
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend
from skimage.morphology import skeletonize

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Hardcoded credentials
BOT_TOKEN = "8054078612:AAFHSQfg27HznxnGYdTVlExIjhYnPXSEiAs"
API_ID = "28298273"    # Replace with your actual API_ID
API_HASH = "34b0c10945d1bd2160c2b76b31f247b7"   # Replace with your actual API_HASH

class LogisticMap:
    def __init__(self, r, initial_condition, iterations):
        self.r = r
        self.x = initial_condition
        self.iterations = iterations

    def generate_sequence(self):
        sequence = np.zeros(self.iterations)
        for i in range(self.iterations):
            self.x = self.r * self.x * (1 - self.x)
            sequence[i] = self.x
        return sequence

class SecureImageBot:
    def __init__(self):
        """Initialize the bot with session management."""
        self.client = TelegramClient('bot_session', API_ID, API_HASH)
        self.private_key = None
        self.public_key = None
        self.user_states = {}  # Track user registration states
        self.pending_encryptions = {}  # Track pending image encryptions
        self.user_fingerprints = {}  # Store fingerprint data securely (simplified)
        self.active_users = set()  # Track users who have started the bot

        # Create required directories with full paths
        self.dirs = {
            'temp': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_files'),
            'fingerprints': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'private', 'fingerprints'),
            'encrypted': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'private', 'encrypted')
        }

        # Create all directories
        for dir_path in self.dirs.values():
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"Error creating directory {dir_path}: {e}")

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
        """Handle private messages and track active users."""
        user_id = event.sender_id

        # Add user to active users when they interact with the bot
        self.active_users.add(user_id)

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
            if user_id not in self.user_fingerprints:
                await event.respond(
                    "⚠️ You need to register first!\n"
                    "Use /register to set up secure messaging."
                )
                return

            self.user_states[user_id] = {
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

            # Process fingerprint (simplified: just count minutiae)
            features = self._extract_fingerprint_features(temp_path)
            if not features:
                await event.respond("❌ Could not process fingerprint. Please send a clearer image.")
                return False

            # Store user's fingerprint data (simplified: just minutiae count)
            self.user_fingerprints[user_id] = {'minutiae_count': len(features)}

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
            "Please forward all 4 encrypted files here.\n"
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
                    # Get recipient entity
                    recipient = await self.client.get_entity(username)

                    # Verify recipient has started the bot
                    if recipient.id not in self.active_users:
                        await event.respond(
                            f"⚠️ {username} needs to start using this bot first!\n\n"
                            f"Please ask them to:\n"
                            f"1. Find this bot: {(await self.client.get_me()).username}\n"
                            f"2. Start a chat and use the /start command\n"
                            f"3. Then try sending the image again"
                        )
                        return

                    # Save recipient info and update state
                    state['recipient'] = recipient
                    state['state'] = 'awaiting_image'
                    await event.respond(
                        "✅ Recipient selected!\n"
                        "Now send the image you want to encrypt and share."
                    )
                except ValueError:
                    await event.respond(
                        "❌ Could not find user.\n"
                        "Please make sure to send their exact Telegram username (with or without @) or forward a message from them."
                    )
            except Exception as e:
                logging.error(f"Error in recipient selection: {e}")
                await event.respond("❌ Error finding recipient. Please try again.")

    async def _handle_media(self, event):
        """Handle media messages in private chat."""
        user_id = event.sender_id
        temp_path = None
        encrypted_files = []

        try:
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

                # Get recipient's stored fingerprint data
                recipient_id = state['recipient'].id
                if recipient_id not in self.user_fingerprints:
                    await event.respond("⚠️ Recipient needs to register their fingerprint first!")
                    return

                recipient_data = self.user_fingerprints[recipient_id]

                self.user_states[user_id]['state'] = 'awaiting_sender_fingerprint'
                self.pending_encryptions[user_id] = {
                    'image_event': event,
                    'recipient_minutiae': recipient_data['minutiae_count']
                }
                await event.respond("Please send your fingerprint image for encryption.")
                return

            elif state.get('state') == 'awaiting_sender_fingerprint':
                if event.photo:
                    sender_fingerprint_path = os.path.join(self.dirs['temp'], f'sender_fp_{user_id}.jpg')
                    await event.download_media(sender_fingerprint_path)
                    sender_features = self._extract_fingerprint_features(sender_fingerprint_path)
                    if not sender_features:
                        await event.respond("❌ Could not process your fingerprint. Please try again.")
                        return

                    # Calculate sender's biometric data
                    sender_bio_key = np.mean([sum(coord) for coord in [(p['x'], p['y']) for p in sender_features]]) / (len(sender_features) + 1e-9)
                    sender_minutiae_count = len(sender_features)
                    
                    # Get recipient's stored fingerprint data
                    recipient_minutiae = self.pending_encryptions[user_id]['recipient_minutiae']
                    
                    # Combine both fingerprint data for encryption
                    combined_bio_key = sender_bio_key  # We use sender's bio key for encryption
                    combined_minutiae = (sender_minutiae_count + recipient_minutiae) // 2

                    image_event = self.pending_encryptions[user_id]['image_event']
                    if image_event and image_event.photo:
                        temp_path = os.path.join(self.dirs['temp'], f'img_{image_event.id}.jpg')
                        await image_event.download_media(temp_path)

                        if os.path.exists(temp_path):
                            try:
                                encrypted_files = await self._encrypt_image(
                                    temp_path,
                                    combined_bio_key,
                                    combined_minutiae
                                )
                                if encrypted_files and state.get('recipient'):
                                    await event.respond("🔒 Sending encrypted files to recipient...")
                                    for file_path in encrypted_files:
                                        await self.client.send_file(
                                            state['recipient'],
                                            file_path,
                                            force_document=True,
                                            caption="🔒 Encrypted image file"
                                        )
                                    await event.respond(
                                        "✅ Encrypted image sent!\n\n"
                                        "Your friend needs to:\n"
                                        "1. Register their fingerprint using /register\n"
                                        "2. Forward all 4 encrypted files to me\n"
                                        "3. Use /decrypt to view the image"
                                    )
                                else:
                                    await event.respond("❌ Encryption failed. Please try again.")
                            except Exception as e:
                                logger.error(f"Encryption error: {e}")
                                await event.respond(f"❌ Failed to encrypt and send image: {str(e)}")
                            finally:
                                # Clean up temporary files
                                if temp_path and os.path.exists(temp_path):
                                    os.remove(temp_path)
                                for file in encrypted_files:
                                    if os.path.exists(file):
                                        os.remove(file)
                                if user_id in self.user_states:
                                    del self.user_states[user_id]
                        else:
                            await event.respond("❌ Error downloading the image to encrypt.")
                    else:
                        await event.respond("❌ Original image not found. Please try again.")
                else:
                    await event.respond("Please send your fingerprint image.")
                return

            # Handle decryption request (receiver)
            elif state.get('state') == 'awaiting_files' and (event.photo or event.document):
                state['files'].append(await event.download_media())
                if len(state['files']) == 4:
                    self.user_states[user_id]['state'] = 'awaiting_sender_fingerprint_decrypt'
                    await event.respond("Please forward or send the fingerprint of the person who sent you this image.")
                return

            elif state.get('state') == 'awaiting_sender_fingerprint_decrypt':
                if event.photo:
                    sender_fingerprint_path = os.path.join(self.dirs['temp'], f'sender_fp_decrypt_{user_id}.jpg')
                    await event.download_media(sender_fingerprint_path)
                    sender_features = self._extract_fingerprint_features(sender_fingerprint_path)
                    if not sender_features:
                        await event.respond("❌ Could not process sender's fingerprint. Please try again.")
                        return

                    # Store sender's fingerprint data for decryption
                    sender_bio_key = np.mean([sum(coord) for coord in [(p['x'], p['y']) for p in sender_features]]) / (len(sender_features) + 1e-9)
                    sender_minutiae_count = len(sender_features)
                    
                    self.user_states[user_id].update({
                        'state': 'awaiting_receiver_fingerprint',
                        'sender_bio_key': sender_bio_key,
                        'sender_minutiae': sender_minutiae_count
                    })
                    await event.respond("Now send your fingerprint (as the receiver) to decrypt the image.")
                else:
                    await event.respond("Please send the sender's fingerprint image.")
                return

            elif state.get('state') == 'awaiting_receiver_fingerprint':
                if event.photo:
                    receiver_fingerprint_path = os.path.join(self.dirs['temp'], f'receiver_fp_{user_id}.jpg')
                    await event.download_media(receiver_fingerprint_path)
                    receiver_features = self._extract_fingerprint_features(receiver_fingerprint_path)
                    if not receiver_features:
                        await event.respond("❌ Could not process your fingerprint. Please try again.")
                        return

                    # Calculate receiver's biometric data
                    receiver_bio_key = np.mean([sum(coord) for coord in [(p['x'], p['y']) for p in receiver_features]]) / (len(receiver_features) + 1e-9)
                    receiver_minutiae_count = len(receiver_features)

                    # Reconstruct the combined key using both fingerprints (same as encryption)
                    combined_bio_key = (state['sender_bio_key'] + receiver_bio_key) / 2
                    combined_minutiae = (state['sender_minutiae'] + receiver_minutiae_count) // 2

                    await event.respond("✅ Both fingerprints verified. Proceeding with decryption...")
                    await self._process_decryption(event, state['files'], combined_bio_key, combined_minutiae)
                    del self.user_states[user_id]  # Clean up state
                else:
                    await event.respond("Please send your fingerprint image to decrypt.")
                return

        except Exception as e:
            logger.error(f"Error handling media: {e}")
            await event.respond("❌ An error occurred while processing your request. Please try again.")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    async def _process_decryption(self, event, files, combined_bio_key, combined_minutiae):
        """Process decryption of received files."""
        output_path = None
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
                logistic_r = float(f.readline().strip())
                logistic_initial_condition = float(f.readline().strip())
                logistic_iterations = int(f.readline().strip())

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

            # Reverse scramble using Logistic Map
            logistic_map = LogisticMap(logistic_r, logistic_initial_condition, logistic_iterations)
            decrypted = self._unscramble_image_with_logistic_map(decrypted_scrambled, permutation, shape)

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
                if output_path and os.path.exists(output_path):
                    os.remove(output_path)
            except Exception as e:
                logger.error(f"Error cleaning up files: {e}")

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

    def _unscramble_image_with_logistic_map(self, scrambled_image, permutation, original_shape):
        """Reverse scramble image using Logistic Map."""
        flattened = scrambled_image.flatten()
        inverse_perm = np.argsort(permutation)
        original = flattened[inverse_perm]
        return original.reshape(original_shape)

    async def _encrypt_image(self, image_path, bio_key, minutiae_count):
        """Encrypt image using hybrid encryption with biometric influence."""
        try:
            # Read and process image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")

            # Generate session key
            aes_key = os.urandom(32)  # 256-bit AES key

            # Derive parameters for Logistic Map from biometric data (simplified)
            logistic_r = 3.99
            logistic_initial_condition = bio_key
            logistic_iterations = minutiae_count * 10

            # Scramble image using Logistic Map
            logistic_map = LogisticMap(logistic_r, logistic_initial_condition, logistic_iterations)
            chaotic_sequence = logistic_map.generate_sequence()
            seed = int(np.sum(chaotic_sequence) * 1000000) % (2**32)
            rng = np.random.RandomState(seed)
            permutation = rng.permutation(image.size)
            scrambled = image.flatten()[permutation].reshape(image.shape)

            # Encrypt with AES
            iv, encrypted_data = self._aes_encrypt(scrambled, aes_key)

            # Encrypt AES key with RSA
            encrypted_key = self._encrypt_aes_key(aes_key)

            # Save encrypted files with timestamp
            timestamp = int(asyncio.get_event_loop().time())
            base_path = os.path.join(self.dirs['encrypted'], f'secure_{timestamp}')

            os.makedirs(os.path.dirname(base_path), exist_ok=True)

            file_paths = []
            try:
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

                # Save metadata including Logistic Map parameters
                meta_path = f"{base_path}_meta.txt"
                with open(meta_path, 'w') as f:
                    f.write(f"{minutiae_count}\n{iv.hex()}\n{image.shape[0]}\n{image.shape[1]}\n{image.shape[2]}\n")
                    f.write(f"{logistic_r}\n{logistic_initial_condition}\n{logistic_iterations}\n")
                file_paths.append(meta_path)

                logger.info(f"Successfully created encrypted files at {base_path}")
                return file_paths

            except Exception as e:
                logger.error(f"Error saving encrypted files: {e}")
                for path in file_paths:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
                raise

        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise

    def _aes_encrypt(self, image, key):
        """Encrypt image using AES."""
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()

        flat = image.flatten()
        pad_size = (len(flat) + 15) // 16 * 16
        padded = np.pad(flat, (0, pad_size - len(flat)), mode='constant')
        encrypted = encryptor.update(padded.tobytes()) + encryptor.finalize()
        
        return iv, encrypted

    def _encrypt_aes_key(self, aes_key):
        """Encrypt AES key using RSA public key."""
        return self.public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

if __name__ == "__main__":
    try:
        # Create bot instance
        bot = SecureImageBot()
        
        # Run the bot using asyncio
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Error running bot: {e}")