import sys
import os
import glob
import pandas as pd
import hdf5_getters
import requests
import re
from DataPreprocessing.preprocess import DataPreprocessor
import musicbrainzngs
from api import LAST_API_KEY


class Song:
    songCount = 0

    # songDictionary = {}

    def __init__(self, songID):
        self.id = songID
        Song.songCount += 1
        # Song.songDictionary[songID] = self

        self.albumName = None
        self.albumID = None
        self.artistID = None
        self.artistLatitude = None
        self.artistLocation = None
        self.artistLongitude = None
        self.artistName = None
        self.danceability = None
        self.duration = None
        self.genreList = []
        self.keySignature = None
        self.keySignatureConfidence = None
        self.lyrics = None
        self.popularity = None
        self.tempo = None
        self.timeSignature = None
        self.timeSignatureConfidence = None
        self.title = None
        self.year = None
        self.genre = None

    def displaySongCount(self):
        print("Total Song Count %i" % Song.songCount)

    def displaySong(self):
        print("ID: %s" % self.id)


def get_lastfm_genres(artist, track):
    url = f'http://ws.audioscrobbler.com/2.0/?method=track.gettoptags&artist={artist}&track={track}&api_key={LAST_API_KEY}&format=json'
    response = requests.get(url)
    if response.status_code == 200:
        tags = response.json().get('toptags', {}).get('tag', [])
        tag_names = [tag['name'] for tag in tags]
        print(tag_names)
        if not tag_names:
            return None
        main_genre = None

        for tag in tag_names:
            if tag.lower() in ['rock', 'pop', 'metal', 'jazz',
                               'classical', 'hip-hop', 'edm', 'dance',
                               'rap', 'r&b', 'reggae', 'alternative', 'indie',
                               'punk', 'folk', 'country', 'electronic']:
                main_genre = tag
        return main_genre
    else:
        print(f"Failed to retrieve tags for {artist} - {track}")
        return None


def main():
    outputFile1 = open('SongCSV.csv', 'w')
    csvRowString = ""

    #################################################
    # change the order of the csv file here
    # Default is to list all available attributes (in alphabetical order)
    csvRowString = ("SongID,AlbumID,AlbumName,ArtistID,ArtistLatitude,ArtistLocation," +
                    "ArtistLongitude,ArtistName,Danceability,Duration,KeySignature," +
                    "KeySignatureConfidence,Tempo,TimeSignature,TimeSignatureConfidence," +
                    "Title,Year,Genre,")
    #################################################
    csvAttributeList = re.split(',', csvRowString)
    for i, v in enumerate(csvAttributeList):
        csvAttributeList[i] = csvAttributeList[i].lower()
    outputFile1.write("SongNumber,")
    outputFile1.write(csvRowString + "\n")
    csvRowString = ""
    #################################################

    # Set the basedir here, the root directory from which the search
    # for files stored in a (hierarchical data structure) will originate
    basedir = "."  # "." As the default means the current directory
    ext = ".H5"  # Set the extension here. H5 is the extension for HDF5 files.
    #################################################

    # FOR LOOP
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root, '*' + ext))
        for f in files:
            print(f)

            songH5File = hdf5_getters.open_h5_file_read(f)
            song = Song(str(hdf5_getters.get_song_id(songH5File)))

            testDanceability = hdf5_getters.get_danceability(songH5File)
            # print type(testDanceability)
            # print ("Here is the danceability: ") + str(testDanceability)

            # Modify the section where you assign values to song attributes
            song.albumID = str(hdf5_getters.get_release_7digitalid(songH5File))[2:-1]
            song.albumName = str(hdf5_getters.get_release(songH5File))[2:-1]
            song.artistID = str(hdf5_getters.get_artist_id(songH5File))[2:-1]
            song.artistLatitude = str(hdf5_getters.get_artist_latitude(songH5File))
            song.artistLocation = str(hdf5_getters.get_artist_location(songH5File))[2:-1]
            song.artistLongitude = str(hdf5_getters.get_artist_longitude(songH5File))
            song.artistName = str(hdf5_getters.get_artist_name(songH5File))[2:-1]
            song.danceability = str(hdf5_getters.get_danceability(songH5File))
            song.duration = str(hdf5_getters.get_duration(songH5File))[2:-1]
            song.keySignature = str(hdf5_getters.get_key(songH5File))
            song.keySignatureConfidence = str(hdf5_getters.get_key_confidence(songH5File))
            song.tempo = str(hdf5_getters.get_tempo(songH5File))
            song.timeSignature = str(hdf5_getters.get_time_signature(songH5File))
            song.timeSignatureConfidence = str(hdf5_getters.get_time_signature_confidence(songH5File))
            song.title = str(hdf5_getters.get_title(songH5File))[2:-1]
            song.year = str(hdf5_getters.get_year(songH5File))
            song.genre = get_lastfm_genres(song.artistName, song.title)

            # print song count
            csvRowString += str(song.songCount) + ","

            for attribute in csvAttributeList:
                # print "Here is the attribute: " + attribute + " \n"
                if attribute == 'AlbumID'.lower():
                    csvRowString += song.albumID
                elif attribute == 'AlbumName'.lower():
                    albumName = song.albumName
                    albumName = albumName.replace(',', "")
                    csvRowString += "\"" + albumName + "\""
                elif attribute == 'ArtistID'.lower():
                    csvRowString += "\"" + song.artistID + "\""
                elif attribute == 'ArtistLatitude'.lower():
                    latitude = song.artistLatitude
                    if latitude == 'nan':
                        latitude = ''
                    csvRowString += latitude
                elif attribute == 'ArtistLocation'.lower():
                    location = song.artistLocation
                    location = location.replace(',', '')
                    csvRowString += "\"" + location + "\""
                elif attribute == 'ArtistLongitude'.lower():
                    longitude = song.artistLongitude
                    if longitude == 'nan':
                        longitude = ''
                    csvRowString += longitude
                elif attribute == 'ArtistName'.lower():
                    csvRowString += "\"" + song.artistName + "\""
                elif attribute == 'Danceability'.lower():
                    csvRowString += song.danceability
                elif attribute == 'Duration'.lower():
                    csvRowString += song.duration
                elif attribute == 'KeySignature'.lower():
                    csvRowString += song.keySignature
                elif attribute == 'KeySignatureConfidence'.lower():
                    # print "key sig conf: " + song.timeSignatureConfidence                                 
                    csvRowString += song.keySignatureConfidence
                elif attribute == 'SongID'.lower():
                    csvRowString += "\"" + song.id[2:-1] + "\""
                elif attribute == 'Tempo'.lower():
                    # print "Tempo: " + song.tempo
                    csvRowString += song.tempo
                elif attribute == 'TimeSignature'.lower():
                    csvRowString += song.timeSignature
                elif attribute == 'TimeSignatureConfidence'.lower():
                    # print "time sig conf: " + song.timeSignatureConfidence                                   
                    csvRowString += song.timeSignatureConfidence
                elif attribute == 'Title'.lower():
                    csvRowString += "\"" + song.title + "\""
                elif attribute == 'Year'.lower():
                    csvRowString += song.year
                elif attribute == 'Genre'.lower() and song.genre:
                    csvRowString += song.genre
                # elif attribute == 'Subgenres'.lower():
                #    csvRowString += ", ".join(song.subgenres)
                #else:
                #    csvRowString += "Erm. This didn't work. Error. :( :(\n"

                csvRowString += ","

            # Remove the final comma from each row in the csv
            lastIndex = len(csvRowString)
            csvRowString = csvRowString[0:lastIndex - 1]
            csvRowString += "\n"
            outputFile1.write(csvRowString)
            csvRowString = ""

            songH5File.close()

    outputFile1.close()

main()
