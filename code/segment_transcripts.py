import os
import re
import nltk
import html2text
import markdown
import numpy as np
import pandas as pd
import brainiak.eventseg.event as event
import hypertools.tools.format_data as fit_transform
from num2words import num2words
from bs4 import BeautifulSoup
from downloader import download_file_from_google_drive as dl

nltk.download('stopwords')
from nltk.corpus import stopwords as sw

data_filepath = '../../data/data.csv'
data_fileID = '1hCCn31z4HM4IzQi59DP-vvUpYKhlvo2S'
# possible numbers of events to try for each transcript
ks_list = list(range(2, 75))

def load_data(filepath, fileid):
    data_dir = os.path.dirname(filepath)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(filepath):
        print('downloading data...')
        dl(fileid, filepath)

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


def get_windows(transcript, wsize=video_wsize):
    cleaned = cleanup_text(wipe_formatting(transcript))
    text_list = cleaned.split('.')
    video_w = []

    for ix, sentence in enumerate(text_list):
        video_w.append(' '.join(text_list[ix:ix+wsize]))

    return video_w


def topic_model(transcript, vec_params=vectorizer_params, sem_params=semantic_params, return_windows=False):
    windows = get_windows(transcript)
    # handle movies with missing or useless transcripts
    if len(windows) < 10:
        return np.nan

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


########################## MAIN SCRIPT ##########################
# download and load data
data_df = load_data(data_filepath, fileid=data_fileID)

# fit topic model, transform script
data_df['trajectory'] = data_df.script.apply(tm_handle_bad)

# drop rows where script content couldn't be modeled
data_df.dropna(inplace=True)

# determine number of events for segmentation
data_df['n_events'] = data_df.trajectory.apply(optimize_k, ks_list=ks_list)

# segment trajectory based on that number
data_df[['segments', 'event onsets/offsets']] = data_df.apply(lambda x:
    segment_trajectory(x['trajectory'], x['n_events']),
    result_type='expand', axis=1)

# save out updated data
data_df.to_csv('../data_modeled.csv')
