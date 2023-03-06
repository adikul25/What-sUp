import streamlit as st
import re
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from empiricaldist import Pmf



def startsWithDate(s):
    #s = s.read()
    pattern = '^([0-2][0-9]|(3)[0-1])(\/)(((0)[0-9])|((1)[0-2]))(\/)(\d{2}|\d{4}), ([0-9][0-9]):([0-9][0-9]) -'
    result = re.match(pattern, s)
    if result:
        return True
    return False

def startsWithAuthor(s):
    patterns = [
        '([\w]+):',                        # First Name
        '([\w]+[\s]+[\w]+):',              # First Name + Last Name
        '([\w]+[\s]+[\w]+[\s]+[\w]+):',    # First Name + Middle Name + Last Name
        '([+]\d{2} \d{5} \d{5}):',         # Mobile Number (India)
        '([+]\d{2} \d{3} \d{3} \d{4}):',   # Mobile Number (US)
        '([+]\d{2} \d{4} \d{7})'           # Mobile Number (Europe)
    ]
    pattern = '^' + '|'.join(patterns)
    result = re.match(pattern, s)
    if result:
        return True
    return False


def getDataPoint(line):
    line = line.encode().decode("utf-8")
    splitLine = line.split(' - ')
    dateTime = splitLine[0]
    date, time = dateTime.split(', ')
    message = ' '.join(splitLine[1:])
    if startsWithAuthor(message):
        splitMessage = message.split(': ')
        author = splitMessage[0]
        message = ' '.join(splitMessage[1:])
    else:
        author = None
    return date, time, author, message

def parse_chat_file(file):
    parsedData = [] # List to keep track of data so it can be used by a Pandas dataframe
    text = file.getvalue().decode("utf-8")

    text = text.split("\n")
    text = [i for i in text if i]  # to remove empty elements

    messageBuffer = [] # Buffer to capture intermediate output for multi-line messages
    date, time, author = None, None, None # Intermediate variables to keep track of the current message being processed
    for line in text:
        line = line.strip() # Guarding against erroneous leading and trailing whitespaces
        if startsWithDate(line): # If a line starts with a Date Time pattern, then this indicates the beginning of a new message
            if len(messageBuffer) > 0: # Check if the message buffer contains characters from previous iterations
                parsedData.append([date, time, author, ' '.join(messageBuffer)]) # Save the tokens from the previous message in parsedData
            messageBuffer.clear() # Clear the message buffer so that it can be used for the next message
            date, time, author, message = getDataPoint(line) # Identify and extract tokens from the line
            messageBuffer.append(message) # Append message to buffer
        else:
            messageBuffer.append(line)
    df = pd.DataFrame(parsedData, columns=["Date", "Time", "Author", "Message"])
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

    return df

st.set_page_config(page_title='WhatsApp Chat Parser')

st.title('WhatsApp Chat Parser')
file = st.file_uploader('Upload a WhatsApp chat file', type=['txt'])

if file is not None:
    df = parse_chat_file(file)
    st.write(df)

    options = ['Total messages', 'Media messages', 'Links shared', 'Most active hours', 'Most active days', 'Author value counts', 'Media messages by author']
    selected_options = st.multiselect('Select analytics to display', options)

    if 'Total messages' in selected_options:
        total_messages = df.shape[0]
        st.write('Total messages:', total_messages)


    if 'Media messages' in selected_options:
        media_messages = df[df['Message'] == '<Media omitted>'].shape[0]
        st.write('Media messages:', media_messages)

    if 'Links shared' in selected_options:
        links_regex = r'http\S+|www\S+'
        links = df['Message'].str.findall(links_regex).sum()
        links_count = len(links)
        st.write('Links shared:', links_count)

    if 'Most active hours' in selected_options:
        df['Hour'] = pd.to_datetime(df['Time']).dt.hour
        active_hours = df.groupby('Hour').size().reset_index(name='Counts')
        st.write('Most active hours:')
        st.bar_chart(active_hours)

    if 'Most active days' in selected_options:
            df['Day'] = df['Date'].dt.weekday_name
            active_days = df.groupby('Day').size().reset_index(name='Counts')
            st.write('Most active days:')
            st.bar_chart(active_days)

    if 'Author value counts' in selected_options:
            author_value_counts = df['Author'].value_counts()
            st.write('Author value counts:')
            st.bar_chart(author_value_counts)

    if 'Media messages by author' in selected_options:
        media_messages_by_author = df[df['Message'] == '<Media omitted>']['Author'].value_counts()
        st.write('Media messages by author:')
        st.bar_chart(media_messages_by_author)
#
# # Author value count
# author_value_count = df['Author'].value_counts()
# st.subheader('Author Value Counts')
# st.write(author_value_count)
#
# # Bar chart of author value count
# st.subheader('Bar Chart of Author Value Counts')
# st.bar_chart(author_value_count)
#
# # Group stats
# total_messages = df.shape[0]
# media_messages = df[df['Message'] == '<Media omitted>'].shape[0]
# URLPATTERN = r'(https?://\S+)'
# df['urlcount'] = df.Message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
# links = np.sum(df.urlcount)
# st.subheader('Group Stats')
# st.write('Messages:', total_messages)
# st.write('Media Messages:', media_messages)
# st.write('URL:', links)
#
# #Media Message
#
# st.subheader('Most Media Messages Sent')
# tot_media_messages = df[df['Message'] == '<Media omitted>']
# most_media_messages = tot_media_messages['Author'].value_counts()
# st.bar_chart(most_media_messages)
