import os
from PIL import Image
from io import BytesIO
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackQueryHandler, CommandHandler
from inference import stylize


TOKEN = 'your token'  # Замените на свой токен


def start(update, context):
    update.message.reply_text('Пожалуйста, отправьте фотографию.')


def process_image(update, context):
    photo = update.message.photo[-1]
    file_id = photo.file_id
    file = context.bot.get_file(file_id)
    image_stream = BytesIO()
    file.download(out=image_stream)
    image_stream.seek(0)

    image = Image.open(image_stream)

    context.user_data['image'] = image

    keyboard = [
        [InlineKeyboardButton("Selfie2Anime (after 50 hours of training)", callback_data='anime')],
        [InlineKeyboardButton("Photo2Mone (after 9 hours of training)", callback_data='photo2mone')],
        [InlineKeyboardButton("Mone2Photo (after 9 hours of training)", callback_data='mone2photo')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Выберите стиль:', reply_markup=reply_markup)


def process_document(update, context):
    document = update.message.document
    file_id = document.file_id
    file = context.bot.get_file(file_id)
    image_stream = BytesIO()
    file.download(out=image_stream)
    image_stream.seek(0)

    if document.mime_type.startswith('image/'):
        image = Image.open(image_stream)

        context.user_data['image'] = image

        keyboard = [
            [InlineKeyboardButton("Selfie2Anime (after 50 hours of training)", callback_data='anime')],
            [InlineKeyboardButton("Photo2Mone (after 9 hours of training)", callback_data='photo2mone')],
            [InlineKeyboardButton("Mone2Photo (after 9 hours of training)", callback_data='mone2photo')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text('Выберите действие:', reply_markup=reply_markup)
    else:
        update.message.reply_text('Отправленный файл не является изображением. Пожалуйста отправьте изображение.')

def process_action(update, context):
    query = update.callback_query
    action = query.data
    image = context.user_data.get('image')

    resized_image = stylize(image, action)

    output_stream = BytesIO()
    resized_image.save(output_stream, format='PNG')
    output_stream.seek(0)

    query.message.reply_photo(photo=output_stream)

    query.message.reply_text('Пожалуйста, отправьте новую фотографию.')

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))

    dp.add_handler(MessageHandler(Filters.photo, process_image))

    dp.add_handler(MessageHandler(Filters.document, process_document))

    dp.add_handler(CallbackQueryHandler(process_action))

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
