import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, filters

# Replace 'YOUR_BOT_TOKEN' with the API token you received from BotFather
TOKEN = '8020166892:AAG_tKMKhvU68oDecn20K7ugaxTV5FWn3dk'

def start(update, context):
    update.message.reply_text('Welcome to the Image Encryption/Decryption Bot!')

def help_command(update, context):
    update.message.reply_text('This bot can encrypt and decrypt images using a hybrid algorithm with biometric authentication.')

def handle_image(update, context):
    # This function will be called when the bot receives an image
    update.message.reply_text('Image received! Now processing...')
    # Here you will integrate your encryption logic

def handle_text(update, context):
    update.message.reply_text('I can only process images. Please send an image to encrypt or decrypt.')

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help_command))
    dp.add_handler(MessageHandler(Filters.photo & ~Filters.command, handle_image))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()