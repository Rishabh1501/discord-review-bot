import os

import discord
from dotenv import load_dotenv
from discord.ext import commands
from predicting_model.prediction import PredictAPI

load_dotenv()

TOKEN = os.environ['TOKEN']
GUILD = os.environ['DISCORD_GUILD']

# client = discord.Client()

client = commands.Bot(command_prefix= '!')

@client.event
async def on_ready():
    for guild in client.guilds:
        if guild.name == GUILD:
            print(
                f'{client.user} is connected to the following guild:\n'
                f'{guild.name}(id: {guild.id})'
            )

@client.command()
async def greet(ctx):
    "Command to greet"
    await ctx.send(f"Hello {ctx.author.mention}")

@client.command()
async def review(ctx,*review):
    """
    Command to predict the Intent of Reviews
    **Don't use quotes(",') in the reviews**
    """
    if "general" == ctx.channel.name:
        await ctx.message.delete()
        await ctx.send(f"{ctx.author.mention},Do not use this command here!")
    else:
        prediction = PredictAPI()
        review = " ".join(review)

        clean_sentence = prediction.clean_sentence(review)
        predicted_data = prediction.predict_model_sentence(clean_sentence)
        await ctx.send(f"It is a {predicted_data} review!")
    
    

@review.error
async def review_error(ctx,error):
    if isinstance(error,discord.ext.commands.errors.InvalidEndOfQuotedStringError):
        await ctx.send("Don't use single or double quotes in the reviews")


@client.event
async def on_message(message):
    author = message.author
    content = message.content
    print(f'{author} = {content}')
    with open("cursewords.txt") as f:
        curse_words = [i.rstrip() for i in f.readlines()]
        for word in content.split():
            if word in curse_words:
                await message.delete()
                await message.channel.send(f"{author.mention}, Don't use curse words, you have been warned!!!")
    await client.process_commands(message)


# @client.event
# async def on_message_delete(message):
#     author = message.author
#     content = message.content
#     channel = message.channel
#     await channel.send(f'{author.mention} ,deleted the message "{content}" ')
    


client.run(TOKEN)