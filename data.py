import pandas as pd

dataset = pd.read_csv('taylor_swift_lyrics.csv', encoding = "latin1")

def processFirstLine(lyrics, songID, songName, row):
    lyrics.append(row['lyric'] + '\n')
    songID.append(row['year']*100 + row['track_n'])
    songName.append(row['track_title'])
    return lyrics, songID, songName

lyrics = []
songID = []
songName =[]

songNumber = 1
i = 0
isFirstLine = True

for index, row in dataset.iterrows():
    if(songNumber == row['track_n']):
        if (isFirstLine):
            lyrics, songID, songName = processFirstLine(lyrics, songID, songName, row)
            isFirstLine = False
        else:
            lyrics[i] += row['lyric'] + '\n'
    else:
        lyrics,songID,songName = processFirstLine(lyrics, songID, songName, row)
        songNumber = row['track_n']
        i+=1

lyrics_data = pd.DataFrame({'songID': songID, 'songName': songName, 'lyrics': lyrics })
with open('lyricsText.txt', 'w', encoding="utf-8") as filehandle:
    for listitem in lyrics:
        filehandle.write('%s\n' % listitem)
