#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A Telegram client that can send and receive images using Telethon.
"""

import os
import logging
import asyncio
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get credentials from environment variables
API_ID = os.getenv('API_ID')
API_HASH = os.getenv('API_HASH')
CHAT_ID = os.getenv('CHAT_ID')

# Directory for sample images
SAMPLE_IMAGES_DIR = "sample_images"
# Directory to save received images
RECEIVED_IMAGES_DIR = "received_images"

# Create directories if they don't exist
os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
os.makedirs(RECEIVED_IMAGES_DIR, exist_ok=True)

# Create the client
client = TelegramClient('telegram_session', API_ID, API_HASH)

async def send_image(chat_id, image_path=None):
    """Send an image to a specified chat ID."""
    try:
        # If no specific image is provided, send a sample image
        if not image_path:
            image_files = [f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            if not image_files:
                logger.error("No sample images available to send.")
                return
            image_path = os.path.join(SAMPLE_IMAGES_DIR, image_files[0])

        # Ensure the image exists
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return

        # Convert chat_id to integer
        chat_id = int(chat_id)

        # Send the image
        await client.send_file(
            chat_id,
            image_path,
            caption="Image sent via Telethon client"
        )

        logger.info(f"Image sent to chat ID: {chat_id}")

    except Exception as e:
        logger.error(f"Error sending image: {e}")

@client.on(events.NewMessage(incoming=True))
async def handle_new_message(event):
    """Handle incoming messages."""
    # Log the message
    logger.info(f"Received message from {event.sender_id}: {event.text}")

    # Check if the message contains an image
    if event.photo or event.document:
        sender = await event.get_sender()
        sender_name = f"{sender.first_name} {sender.last_name if sender.last_name else ''}"

        # Download the image
        if event.photo:
            file_path = os.path.join(RECEIVED_IMAGES_DIR, f"{sender.id}_{event.photo.id}.jpg")
            await event.download_media(file_path)
            logger.info(f"Image received from {sender_name} (ID: {sender.id}) and saved to {file_path}")
            await event.reply(f"Thanks for the image! I've saved it.")
        elif event.document and event.document.mime_type.startswith('image/'):
            file_path = os.path.join(RECEIVED_IMAGES_DIR, f"{sender.id}_{event.document.id}{os.path.splitext(event.document.attributes[0].file_name)[1]}")
            await event.download_media(file_path)
            logger.info(f"Image document received from {sender_name} (ID: {sender.id}) and saved to {file_path}")
            await event.reply(f"Thanks for the image document! I've saved it.")

async def main():
    """Main function to run the client."""
    # Start the client
    print("Starting Telegram client...")
    print("You will need to authenticate with your phone number.")
    print("This is only required once, and the session will be saved.")

    # Phone number to use for authentication
    phone = input("Enter your phone number (with country code, e.g., +1234567890): ")

    try:
        # Start the client and connect
        await client.connect()

        # Check if already authorized
        if not await client.is_user_authorized():
            logger.info("Sending code request...")
            await client.send_code_request(phone)

            # Ask for the verification code
            code = input("Enter the code you received: ")

            try:
                # Sign in with the code
                await client.sign_in(phone, code)
            except SessionPasswordNeededError:
                # 2FA is enabled
                password = input("Enter your 2FA password: ")
                await client.sign_in(password=password)

        logger.info("Successfully logged in!")

        # Send a sample image to the default chat ID
        if CHAT_ID:
            logger.info(f"Sending image to chat ID: {CHAT_ID}")
            await send_image(CHAT_ID)
            logger.info("Image sent successfully!")

        # Keep the client running
        logger.info("Client is now running. Press Ctrl+C to stop.")
        await client.run_until_disconnected()

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == '__main__':
    # Run the main function
    asyncio.run(main())
