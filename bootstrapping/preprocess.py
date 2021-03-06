#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2016 Pascal Jürgens and Andreas Jungherr
# See License.txt

"""
Examples for accessing the API
------------------------------
These are some examples demonstrating the use of provided functions for
gathering data from the Twitter API.

Requirements:
    - depends on API access modules rest.py and streaming.py
"""

import re
import sys
import rest
import streaming
import database
import logging
import json
import datetime
from pytz import timezone
import peewee
from progress.bar import Bar
import nltk
from nltk.corpus import gutenberg
# from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from pprint import pprint

EST = timezone("EST")

#
# Helper Functions
#

# Hashtags
hash_regex = re.compile(r"#(\w+)")
def hash_repl(match):
	return '__HASH_'+match.group(1).upper()

# Handels
hndl_regex = re.compile(r"@(\w+)")
def hndl_repl(match):
	return '__HNDL'#_'+match.group(1).upper()

# URLs
url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")

# Spliting by word boundaries
word_bound_regex = re.compile(r"\W+")

# Repeating words like hurrrryyyyyy
rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE);
def rpt_repl(match):
	return match.group(1)+match.group(1)

# Emoticons
emoticons = \
	[	('__EMOT_SMILEY',	[':-)', ':)', '(:', '(-:', ] )	,\
		('__EMOT_LAUGH',		[':-D', ':D', 'X-D', 'XD', 'xD', ] )	,\
		('__EMOT_LOVE',		['<3', ':\*', ] )	,\
		('__EMOT_WINK',		[';-)', ';)', ';-D', ';D', '(;', '(-;', ] )	,\
		('__EMOT_FROWN',		[':-(', ':(', '(:', '(-:', ] )	,\
		('__EMOT_CRY',		[':,(', ':\'(', ':"(', ':(('] )	,\
	]

# Punctuations
punctuations = \
	[	#('',		['.', ] )	,\
		#('',		[',', ] )	,\
		#('',		['\'', '\"', ] )	,\
		('__PUNC_EXCL',		['!', '¡', ] )	,\
		('__PUNC_QUES',		['?', '¿', ] )	,\
		('__PUNC_ELLP',		['...', '…', ] )	,\
		#FIXME : MORE? http://en.wikipedia.org/wiki/Punctuation
	]

def print_config(cfg):
    for (x, arr) in cfg:
        print(x, '\t')
        for a in arr:
            print(a, '\t')
        print('')

def print_emoticons():
	print_config(emoticons)

def print_punctuations():
	print_config(punctuations)

#For emoticon regexes
def escape_paren(arr):
	return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

def regex_union(arr):
	return '(' + '|'.join( arr ) + ')'

emoticons_regex = [ (repl, re.compile(regex_union(escape_paren(regx))) ) \
					for (repl, regx) in emoticons ]

#For punctuation replacement
def punctuations_repl(match):
	text = match.group(0)
	repl = []
	for (key, parr) in punctuations :
		for punc in parr :
			if punc in text:
				repl.append(key)
	if( len(repl)>0 ) :
		return ' '+' '.join(repl)+' '
	else :
		return ' '

def processHashtags(text, subject='', query=[]):
	return re.sub( hash_regex, hash_repl, text )

def processHandles(text, subject='', query=[]):
	return re.sub( hndl_regex, hndl_repl, text )

def processUrls(text, subject='', query=[]):
	return re.sub( url_regex, ' __URL ', text )

def processEmoticons(text, subject='', query=[]):
	for (repl, regx) in emoticons_regex :
		text = re.sub(regx, ' '+repl+' ', text)
	return text

def processPunctuations( text, subject='', query=[]):
	return re.sub( word_bound_regex , punctuations_repl, text )

def processRepeatings( 	text, subject='', query=[]):
	return re.sub( rpt_regex, rpt_repl, text )

def processQueryTerm( 	text, subject='', query=[]):
	query_regex = "|".join([ re.escape(q) for q in query])
	return re.sub( query_regex, '__QUER', text, flags=re.IGNORECASE )

def countHandles(text):
	return len( re.findall( hndl_regex, text) )

def countHashtags(text):
	return len( re.findall( hash_regex, text) )

def countUrls(text):
	return len( re.findall( url_regex, text) )

def countEmoticons(text):
	count = 0
	for (repl, regx) in emoticons_regex :
		count += len( re.findall( regx, text) )
	return count

#    
# Setup
#

def hydrate(idlist_file="data/example_dataset_tweet_ids.txt"):
    """
    This function reads a file with tweet IDs and then loads them
    through the API into the database. Prepare to wait quite a bit,
    depending on the size of the dataset.
    """
    ids_to_fetch = set()
    for line in open(idlist_file, "r"):
        # Remove newline character through .strip()
        # Convert to int since that's what the database uses
        ids_to_fetch.add(int(line.strip()))
    # Find a list of Tweets that we already have
    ids_in_db = set(t.id for t in database.Tweet.select(database.Tweet.id))
    # Sets have an efficient .difference() method that returns IDs only present
    # in the first set, but not in the second.
    ids_to_fetch = ids_to_fetch.difference(ids_in_db)
    logging.warning(
        "\nLoaded a list of {0} tweet IDs to hydrate".format(len(ids_to_fetch)))

    # Set up a progressbar
    bar = Bar('Fetching tweets', max=len(ids_to_fetch), suffix='%(eta)ds')
    for page in rest.fetch_tweet_list(ids_to_fetch):
        bar.next(len(page))
        for tweet in page:
            database.create_tweet_from_dict(tweet)
    bar.finish()
    logging.warning("Done hydrating!")


def dehydrate(filename="data/dehydrated_tweet_ids.txt"):
    """
    This function writes the Tweet IDs contained in the current database to
    a file that allows re-hydration with the above method.
    """
    with open(filename, "w") as f:
        for tweet in database.Tweet.select(database.Tweet.id):
            f.write("{0}\n".format(tweet.id))


#
# Helper Functions
#

def print_tweet(tweet):
    """
    Print a tweet as one line:
    user: tweet
    """
    logging.info(u"{0}: {1}".format(tweet["user"]["screen_name"], tweet["text"]))


def print_notice(notice):
    """
    This just prints the raw response, such as:
    {u'track': 1, u'timestamp_ms': u'1446089368786'}}
    """
    logging.error(u"{0}".format(notice))

def porter_stemming(raw_word):
    """
    Utilize Porter Stemmer to normalize words
    """
    porter_stemmer = PorterStemmer()
    after_word = porter_stemmer.stem(raw_word.lower())
    return after_word

def wordnet_lemm(raw_word):
    """
    Utitlize WordNet lemmatizer 
    """
    # >>> from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    after_word = wordnet_lemmatizer.lemmatize(raw_word)
    return after_word

def cleanse_sentence(text):
    """
    Cleansing tweet text (sentence) and remove hashtag, handle and url 
    Replace emotions with clear text
    """
    
    text = re.sub(hash_regex, hash_repl, text)
    text = re.sub(hndl_regex, hndl_repl, text)
    text = re.sub(url_regex, ' __URL ', text)
    
    for (repl, regx) in emoticons_regex :
        text = re.sub(regx, ' '+repl+' ', text)

    text = text.replace('\'','')
    text = re.sub( word_bound_regex , punctuations_repl, text )
    text = re.sub( rpt_regex, rpt_repl, text )
    
    return text

def filter_stop_words(processed_word_list, current_tweet_text):
    """
    Remove stopwords so we have concentrated keywords to analyze
    """
    stops = set(stopwords.words("english"))
    word_list = word_tokenize(current_tweet_text)
    # filtered_word_list = word_tokenize(current_tweet_text)

    for word in word_list: # iterate over word_list
        word_stem = porter_stemming(word)
        word_lemm = wordnet_lemm(word_stem)
        if word_lemm in stops: 
            print('Stopword caught ', word_stem) # DEBUG
            # filtered_word_list.remove(word) 
        else:
            processed_word_list.append(word_lemm)
    return processed_word_list

def plot_tweets_by_wordlist(filtered_word_list, bCumulative):
    """
    Show frequency of all words in a given tweet text
    """
    freq = nltk.FreqDist(filtered_word_list)

    for key,val in freq.items():
        print (str(key) + ':' + str(val))
    freq.plot(50, cumulative = bCumulative)

# Main
def main():
    archive = rest.fetch_user_archive(sys.argv[1])
    processed_word_list = []

    for page in archive:
        for tweet in page:
            # print_tweet(tweet)
            cur_text = tweet["text"]
            # print(u"{0}: {1}".format(tweet["user"]["screen_name"], cur_text))
            # print("Length of the tweets: ", len(cur_text))
            cur_text = cleanse_sentence(cur_text)
            processed_word_list = filter_stop_words(processed_word_list, cur_text)
            plot_tweets_by_wordlist(processed_word_list, True)
        break

main()
