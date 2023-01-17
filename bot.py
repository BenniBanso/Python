#import discord
#from discord.ext import commands
import random
from datetime import date

client = commands.Bot(command_prefix="!", intents=discord.Intents.all())

@client.event
async def on_ready():
    print("Bot is connected to Discord")
    

@client.command()
async def ping(ctx):
    await ctx.send("Pong!")

global PlayerList
global ItemList
ItemList = ["MC_BLOCK1", "MC_BLOCK2", "MC_BLOCK3", "MC_BLOCK4", "MC_BLOCK5", "MC_BLOCK6", "MC_BLOCK7", "MC_BLOCK8", "MC_BLOCK9"]
PlayerList= []
RandomCalender = {}

class Playername:
    
    @client.command
    async def player_name(self, ctx):
        self.name = ctx.message.author.display_name

    def player_name_to_list(self, ctx): 
        return PlayerList.append(self.name)

    def Player_Calender(self):
        self.playerCalender = {}
        for i in range(24):
            self.playerCalender[str(i)] = ItemList[random.randint(0, len(ItemList)-1)]
        return self.playerCalender
    
    def get_current_Date(self):
        self.current_day = date.today().day
        return self.current_day
    
    def getPresent(self):
        if str(self.current_day) in self.playerCalender.keys():
            #Mc: give player item
            self.playerCalender.pop(str(self.current_day))
        else:
            #return following message: Du hast das das Geschenk heute schon angenommen!
            pass
        return self.playerCalender


Player = Playername()
print(PlayerList)
print(Player.player_name(), "=", Player.Player_Calender())
print(Player.get_current_Date())
print(Player.getPresent())

client.run("YOUR_KEY")