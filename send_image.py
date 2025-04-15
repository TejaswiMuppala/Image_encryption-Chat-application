#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to send images via Telegram using Telethon.
"""

import os
import logging
import asyncio
import argparse
from dotenv import load_dotenv
from telethon import TelegramClient
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
DEFAULT_CHAT_ID = os.getenv('CHAT_ID')

# Directory for sample images
SAMPLE_IMAGES_DIR = "sample_images"
os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)

async def send_image(client, chat_id, image_path=None):
    """Send an image to a specified chat ID."""
    try:
        # If no specific image is provided, send a sample image
        if not image_path:
            image_files = [f for f in os.listdir(SAMPLE_IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.avif'))]
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

async def main():
    """Main function to parse arguments and send images."""
    parser = argparse.ArgumentParser(description='Send images via Telegram')
    parser.add_argument('--chat_id', type=str, help='Chat ID to send the image to (defaults to CHAT_ID from .env)')
    parser.add_argument('--image', type=str, help='Path to the image to send (defaults to first image in sample_images)')
    parser.add_argument('--phone', type=str, help='Phone number for authentication (with country code)')

    args = parser.parse_args()

    # Use default chat ID if none provided
    chat_id = args.chat_id if args.chat_id else DEFAULT_CHAT_ID
    if not chat_id:
        logger.error("No chat ID provided and no default CHAT_ID in .env")
        return

    # Create the client
    client = TelegramClient('telegram_session', API_ID, API_HASH)

    try:
        # Connect to Telegram
        await client.connect()

        # Check if already authorized
        if not await client.is_user_authorized():
            # Get phone number
            phone = args.phone if args.phone else input("Enter your phone number (with country code, e.g., +1234567890): ")

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

        # Send the image
        await send_image(client, chat_id, args.image)

        # Disconnect the client
        await client.disconnect()

    except Exception as e:
        logger.error(f"Error: {e}")
        # Ensure client is disconnected
        try:
            await client.disconnect()
        except:
            pass

if __name__ == '__main__':
    # Run the main function
    asyncio.run(main())
