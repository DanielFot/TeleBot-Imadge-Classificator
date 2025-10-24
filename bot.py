import tensorflow as tf
from tensorflow import keras
import logging
import numpy as np
from PIL import Image
from io import BytesIO
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
from telegram import Update
from telegram.ext.callbackcontext import CallbackContext
import tensorflow as tf
from tensorflow import keras

# --- Patch DepthwiseConv2D (Teachable-Machine .h5 quirk) ---
orig_init = keras.layers.DepthwiseConv2D.__init__
def patched_init(self, *args, **kwargs):
    kwargs.pop("groups", None)
    orig_init(self, *args, **kwargs)
keras.layers.DepthwiseConv2D.__init__ = patched_init

TOKEN = "8244317619:AAEwLcj1-Q83cuYP9Ead686g2eNCJNXZibg"
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
IMG_SIZE = (224, 224)

logging.basicConfig(level=logging.INFO)
model = keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    class_names = [line.strip() for line in f]

def predict_image(image):
    image = image.convert("RGB").resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.expand_dims(np.asarray(image), 0)
    arr = (arr.astype(np.float32) / 127.5) - 1
    preds = model.predict(arr)[0]
    results = sorted(zip(class_names, preds), key=lambda x: x[1], reverse=True)
    return results

def start(update, context):
    update.message.reply_text("🐾 Send me a photo of an animal!")

def handle_image(update, context):
    photo = update.message.photo[-1].get_file()
    img = Image.open(BytesIO(photo.download_as_bytearray()))
    update.message.reply_text("🔍 Analyzing...")
    results = predict_image(img)
    msg = "📊 *Prediction Results:*\n" + "\n".join(f"• {n}: {s*100:.2f}%" for n, s in results)
    update.message.reply_text(msg, parse_mode="Markdown")

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, handle_image))
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()