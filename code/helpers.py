import os, re, nltk, html2text, markdown, requests
import numpy as np
import pandas as pd
import brainiak.eventseg.event as event
import hypertools.tools.format_data as fit_transform
from num2words import num2words
from bs4 import BeautifulSoup
from imdb import IMDb

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def load_data(filepath, fileid):
    data_dir = os.path.dirname(filepath)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(filepath):
        print('downloading data...')
        download_file_from_google_drive(fileid, filepath)

    print('loading data...')
    data = pd.read_csv(filepath)
    return data.dropna(subset=['script'])


def wipe_formatting(script, rehtml=False):
    parser = html2text.HTML2Text()
    parser.wrap_links = True
    parser.skip_internal_links = True
    parser.inline_links = True
    parser.ignore_anchors = True
    parser.ignore_images = True
    parser.ignore_emphasis = True
    parser.ignore_links = True
    text = parser.handle(script)
    text = text.strip(' \t\n\r')
    if rehtml:
        text = text.replace('\n', '<br/>')
        text = text.replace('\\', '')
    md = markdown.markdown(text)
    soup=BeautifulSoup(md,'html5lib')
    soup=soup.get_text()
    soup = soup.replace('\n', ' ')
    return soup

def cleanup_text(transcript):
    lower_nopunc = re.sub("[^\w\s.]+", '', transcript.lower())    # remove all punctuation except periods (deliminers)
    no_digit = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), lower_nopunc)    # convert digits to words
    spaced = ' '.join(no_digit.replace(',', ' ').split())    # deal with inconsistent whitespace
    return spaced


def wipe_acting_instructions_from_dialogue(line):
    # In many scripts, pieces of dialogue will start with acting instructions in parentheses. For example:
    #       JOHN
    #   (whincing in pain)
    #   Ouch!
    #
    # This function erases those acting instructions, so that only the true spoken dialogue remains.
    if len(re.findall('\([\w\s]+\)',line)) > 0:
        start, end = re.search('\([\w\s+]+\)',line).span()
        line = line[0:start] + line[end+1:]
    return line


def cleanup_text(transcript):
    lower_nopunc = re.sub("[^\w\s.]+", '', transcript.lower())    # remove all punctuation except periods (deliminers)
    no_digit = re.sub(r"(\d+)", lambda x: num2words(int(x.group(0))), lower_nopunc)    # convert digits to words
    spaced = ' '.join(no_digit.replace(',', ' ').split())    # deal with inconsistent whitespace
    return spaced


def get_windows(transcript, wsize = 50):
    cleaned = cleanup_text(wipe_formatting(transcript))
    text_list = cleaned.split('.')
    video_w = []

    for ix, sentence in enumerate(text_list):
        video_w.append(' '.join(text_list[ix:ix+wsize]))

    return video_w


def topic_model(transcript, vec_params = None, sem_params = None, return_windows=False):
    windows = get_windows(transcript)
    # handle movies with missing or useless transcripts
    if len(windows) < 10:
        return np.nan

    if not vec_params:
        vec_params = {
            'model' : 'CountVectorizer',
            'params' : {
                'stop_words' : sw.words('english')
            }
        }

    if not sem_params:
        sem_params = {
            'model' : 'LatentDirichletAllocation',
            'params' : {
                'n_components' : 100,
                'learning_method' : 'batch',
                'random_state' : 0,
            }
        }

    traj = fit_transform(windows, vectorizer=vec_params, semantic=sem_params, corpus=windows)[0]

    if return_windows:
        return traj, windows
    else:
        return traj


def tm_handle_bad(transcript, kwargs):
    try:
        return topic_model(transcript, **kwargs)
    # just catch all errors, nearly all data passes
    # often IndexError due to script being all newlines
    except:
        return np.nan


def optimize_k(trajectory, ks_list):
    corrmat = np.corrcoef(trajectory)
    scores = []

    for k in ks_list:
        ev = event.EventSegment(k)
        ev.fit(trajectory)
        w = np.round(ev.segments_[0]).astype(int)
        mask = np.sum(list(map(lambda x: np.outer(x, x), w.T)), 0).astype(bool)
        within = corrmat[mask].mean()
        across = corrmat[~mask].mean()
        scores.append((within, across))

    t = list(map(lambda x: x[0]/(x[1]-(scores[0][0]/scores[0][1])), scores))
    t /= np.max(t)
    ratios = list(map(lambda x: x - k/(5*video_wsize), t))
    return ks_list[np.argmax(ratios)]


def segment_trajectory(traj, k):
    ev = event.EventSegment(k)
    ev.fit(traj)
    w = (np.round(ev.segments_[0])==1).astype(bool)
    segs = np.array([traj[wi, :].mean(0) for wi in w.T])
    event_times = []
    for s in ev.segments_[0].T:
        tp = np.where(np.round(s)==1)[0]
        event_times.append((tp[0], tp[-1]))

    return segs, event_times


def tm_handle_bad(transcript, **kwargs):
    try:
        return topic_model(transcript, **kwargs)
    # just catch all errors, nearly all data passes
    # often IndexError due to script being all newlines
    except:
        return np.nan
<<<<<<< HEAD
=======

def get_actor_char(mov):

    actors = mov['cast'] # get actors
    act = list()
    char = list()
    mov_name = list()
    # loop through actors and get their respective chars
    for a in actors:
        if len(a.currentRole) > 1:
            tmp_role = a.currentRole
            for role in tmp_role:
                if role.has_key('name'):
                    act.append(a['name'])
                    char.append(role['name'])
                    mov_name.append(mov['title'])
        else:
            if a.currentRole.has_key('name'):
                    act.append(a['name'])
                    char.append(a.currentRole['name'])
                    mov_name.append(mov['title'])

    return(act, char, mov_name)

def fetch_movie_info(moviename):
    ia = IMDb()
    # search the imdb data base for the anme of the movie
    # grab the first movie and retrieve its movie id then fetch movie data
    movies=ia.search_movie(moviename, results=10, _episodes=False)
    tmp_id=movies[0].movieID
    tmp_movie = ia.get_movie(tmp_id)
    ia.update(tmp_movie,info = ['reviews'])

    return(tmp_movie)

def get_reviews(mov):

    review=list()
    rating=list()
    mov_name=list()

    for m in mov['reviews']:
        mov_name.append(mov['title'])
        if len(z['content']) >= 1:
            review.append(z['content'])
            if type(z['rating']) is not None:
                mov_name.append(z['rating'])
            else:
                mov_name.append('None')

    return(review,rating,mov_name)
>>>>>>> 149b50bf999af1acce9e640b018f179dac2a3e56
