import asyncio
import logging
import sys
from time import time
from translate_m2m import translate
from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
# import warnings

# warnings.filterwarnings("ignore")
TOKEN = "8197895153:AAGjmGMJoKDcZZgB_tJZddmC315bU4cDdaE"
time_counter = 0
msgs_num = 0
dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Hello, {html.bold(message.from_user.full_name)}!")


@dp.message()
async def echo_handler(message: Message) -> None:
    global time_counter, msgs_num
    try:
        await message.answer("Ваш запрос обрабатывается. Подождите...")
        st = time()
        a = translate(message.model_copy().text)[0]
        print(message.model_copy().text, a)
        await message.answer(a)
        nd = time()
        time_counter += (nd - st)
        msgs_num += 1
        print(time_counter / msgs_num)
    except Exception as e:
        print(e)
        await message.answer("Ошибка! Попробуйте ещё раз позже и напишите, пожалуйста, об этой ошибке моему создателю(@cheslav_pet).")


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())