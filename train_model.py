#!/usr/bin/env python
# coding: utf-8

import os
import json
import re
import asteval  #alt: from ast import literal_eval
import pandas as pd
import numpy as np
import spacy
import gensim
import gensim.corpora as corpora
from gensim.test.utils import datapath
import pyLDAvis
import pyLDAvis.gensim_models
from nltk.tokenize import word_tokenize
import tensorflow as tf
from keras.utils import pad_sequences
from matplotlib import pyplot as plt
from reinforce import rrun, cluster_verbs, calc_rewards
from tqdm import tqdm
tqdm.pandas()

def import_data(data_folder, import_from):
    print("Importing datasets...")
    aeval = asteval.Interpreter()
    plots = pd.read_csv(os.path.join(data_folder, import_from, 'plots.csv'), converters={'tokens': aeval, 'events': aeval, 'events_generalised': aeval}, encoding='utf-8')
    plots.set_index(['story'], inplace=True)
    metadata = pd.read_csv(os.path.join(data_folder, import_from, 'metadata.csv'), converters={'genres': aeval, 'entities':aeval}, encoding='utf-8')
    metadata.set_index(['story'], inplace=True)
    return plots, metadata

# FUNCTION:
#   use Latent Dirichet Allocation to assign a cluster number to each story based on genre and content
# OPTIONS:
#   genre_clusters: no. of clusters/topics
def cluster_genres(data_folder, plots, data, genre_clusters=10, use_case="train"):
    print("Preparing data for LDA topic clustering...")
    #1. prepare and clean data for LDA
        #1.1. join lists of words back into summaries
    plots['summary'] = plots['tokens'].map(lambda ts: " ".join(ts)) #words aggregated into sentences
    plot_summaries = plots.groupby('story').agg({'summary': lambda x: ' '.join(x)}) #sentences agg'd into full summaries
    plots.drop(columns='summary', inplace=True)

        #1.2. clean summaries
    plot_summaries['summary'] = plot_summaries['summary'].map(
        lambda x: re.sub(r"[$¢£€¥₡₱₽₩]", 'money', x))                # replace currencies with 'money'                       
    plot_summaries['summary'] = plot_summaries['summary'].map(
        lambda x: re.sub(r"[^A-Za-z'’ ]+", '', x))                   # remove all numbers, punctuation except '‘’
    plot_summaries['summary'] = plot_summaries['summary'].map(
        lambda x: re.sub(r"\s*[A-Z]\w*\s*", ' ', x))                #remove all title case words (attempt to remove names)
    plot_summaries['summary'] = plot_summaries['summary'].map(
        lambda x: re.sub(r"ly\s", ' ', x))                           # remove all 'ly' endings (convert adverbs) 
    # note: not all correct, as not all adverbs ((fami)|(ho)|(f)|(sil)|(bul)|(al)|(li)|(re)|(monopo)|(supp)|(ral)|(ug)|...), 
    # but this still functions correctly/same as changed words remain unique
    plot_summaries['summary'] =  plot_summaries['summary'].str.lower()

        #1.3. tokenise remaining words
    plot_summaries['tokens'] = plot_summaries['summary'].map(word_tokenize)

        #1.4. remove stopwords
    sp = spacy.load('en_core_web_sm')
    stopwords = sp.Defaults.stop_words
    extra_stopwords = pd.read_csv(os.path.join(data_folder, 'genre_clustering\\extra_stopwords.csv'), header=None)
    extra_stopwords = extra_stopwords[0].values.tolist()
    stopwords.update(extra_stopwords)
    plot_summaries['tokens'] = plot_summaries['tokens'].map(
        lambda x: [token for token in x if token not in stopwords])

        #1.5. incorporate known genre information too (current implementation: add as extra tokens)
    plot_summaries['tokens'] = plot_summaries['tokens'] + data['genres']
    
        #1.6. convert words to ids and frequencies
    id2word = corpora.Dictionary(plot_summaries['tokens'])
    plot_summaries['word_freq'] = plot_summaries['tokens'].map(lambda x: id2word.doc2bow(x))
    
    #2. perform LDA on plot_summaries
    # parameters can be found here: https://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore
    corpus = plot_summaries['word_freq']
    if use_case == "train":
        print("Doing LDA...") 
        lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=genre_clusters)   #runs long but no tqdm as multiprocessing
    elif use_case == "test":
        print("Applying LDA model...") 
        lda_model = gensim.models.ldamodel.LdaModel.load(os.path.join(data_folder, 'genre_clustering\\lda_model'))
    doc_lda = lda_model[corpus]
    topics = [sorted(doc_lda[i], key=lambda tup: tup[1], reverse=True)[0][0] 
                for i in range(len(doc_lda))] #sort by percentage descending
    data['topic'] = topics # attach genre cluster to entries in data

    #3. visualise LDA
    if use_case == "train":
        print("Visualising LDA results...")
        genre_data_folder = os.path.join(data_folder, 'genre_clustering')
        LDAvis_prepared = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)   
        pyLDAvis.save_html(LDAvis_prepared, os.path.join(genre_data_folder, 'ldavis_'+ str(genre_clusters) +'.html'))
        print("Saved LDA model visualisation to:", genre_data_folder)
        # save lda_model for fast reload/testing! --> load with: lda = gensim.models.ldamodel.LdaModel.load(path)
        lda_model.save(datapath(os.path.join(genre_data_folder, 'lda_model')))
        print("Saved LDA model.")
    print("Topic cluster numbers added to data.")

    return data

# FUNCTION: 
#   encode data for input into neural network
# OPTIONS:
#   genre_clusters: no. of clusters/topics (=0 if you don't want to add LDA genre information)
def encode_train(data_folder, plots, metadata, genre_clusters, generalised):
    #1. transform data into flat summaries and topic ids only
    print("Transforming data for encoding...")
    story_nums = list(plots.index.unique())
    data = pd.DataFrame(story_nums, columns=['story'])
    data['genres'] = metadata['genres']
    data['genres'] = data['genres'].apply(lambda d: d if isinstance(d, list) else [])   #not all story_nums have values in metadata
    data.set_index(['story'], inplace=True)
    if generalised:
        events_flat = [ [e for event_line in plots.loc[story_num, 'events_generalised'] for e in event_line] for story_num in story_nums ]
    else:
        events_flat = [ [e for event_line in plots.loc[story_num, 'events'] for e in event_line] for story_num in story_nums ]
    data['events'] = events_flat

    # fix None events (something went wrong [usually very few items in import])
    fix_df = pd.DataFrame(story_nums, columns=['story'])
    fix_df['n_dropped'] = data['events'].map(lambda plot: np.sum([1 for event in plot if event is None]))
    fix_df['plot_length'] = data['events'].map(len)
    drop_nums = []
    for story_num in story_nums:
        #if more than 20% of events in some singular plot are dropped, drop it
        if fix_df['n_dropped'].loc[story_num] > 0.20 * fix_df['plot_length'].loc[story_num]:   
            drop_nums.append(story_num)
    data.drop(drop_nums, inplace=True)
    story_nums = list(data.index.unique()) #update story_nums
    data['events'] = data['events'].map(lambda plot: [event for event in plot if event is not None])
    n_dropped = int(np.sum(fix_df['n_dropped']))
    if n_dropped > 0:
        print(n_dropped, "events were dropped due to import issues (" + str(n_dropped / np.sum(fix_df['plot_length'])) + "%% of total), and", len(drop_nums), "stories were dropped.")

    #2. (OPTIONAL) add genre cluster (data['topic']:int)
    if (genre_clusters > 0):
        data = cluster_genres(data_folder, plots, data, genre_clusters)

    #3.1. encode genres numerically (one-hot)
    print("Encoding genre data...")
    all_genres = list(pd.Series([x for item in data['genres'] for x in item]).unique())
    genres_idxs = {g:id for id, g in enumerate(all_genres)}
    def encode_genres(ls):
        vec = np.zeros(len(all_genres), dtype=int)
        for g in ls:
            vec[ genres_idxs[g] ] = 1
        return vec
    data['genres'] = data['genres'].progress_map(lambda ls: encode_genres(ls))

    #3.2. encode events
    print("Encoding event data...")
    all_words = pd.Series([w for plot_events in data['events'] for event in plot_events for w in event])
    all_words = list(pd.concat([pd.Series(["<START>", "<END>", "<EMPTY>", None]), all_words], ignore_index=True).unique())
    words_ids = {w:id for id, w in enumerate(all_words)}
    def encode_event(e):
        return np.asarray([words_ids[w] for w in e])
    data['events'] = data['events'].progress_map(lambda ls: [encode_event(e) for e in ls])

    #3.2.1. drop all mishapen events then reshape (stack) events
    drop_nums = []
    stacked_events = {} 
    for story_num in story_nums:
        try:
            if (np.shape(data['events'].loc[story_num])[1] != 5):           #ensure grouped correctly
                raise Exception
            stacked_events[story_num] = np.stack(data['events'].loc[story_num])
        except:
            drop_nums.append(story_num) #if plot is mishapen, drop it
    data.drop(drop_nums, inplace=True)
    story_nums = list(data.index.unique())
    if int(np.sum(drop_nums)) > 0:
        print(len(drop_nums), "stories were dropped due to broken shapes.")
    #data['events'] = data['events'].map(lambda x: np.stack(x))
    data['events'] = pd.Series(stacked_events)

    #3.2.2. pad events: add <START> (0) and <END> (1) tokens to sequences, then pad with <EMPTY> (2)
    print("Padding event sequences...")
    data['events'] = data['events'].progress_map(lambda ls: np.concatenate(([[0,0,0,0,0]], ls, [[1,1,1,1,1]])))
    data['events'] = list(pad_sequences(data['events'], padding='post', value=np.full(5, 2, dtype=int)))   #pad event sequences with empty events [2,2,2,2,2]

    #3.3. calculate frequencies of each word
    print("Calculating word frequencies...")
    subj_freqs = {}
    verb_freqs = {}
    obj_freqs = {}
    prep_freqs = {}
    mod_freqs = {}
    def add_to_freqs(e):
        if not e[0] in subj_freqs: subj_freqs[e[0]] = 1
        else: subj_freqs[e[0]] += 1
        if not e[1] in verb_freqs: verb_freqs[e[1]] = 1
        else: verb_freqs[e[1]] += 1
        if not e[2] in obj_freqs: obj_freqs[e[2]] = 1
        else: obj_freqs[e[2]] += 1
        if not e[3] in prep_freqs: prep_freqs[e[3]] = 1
        else: prep_freqs[e[3]] += 1
        if not e[4] in mod_freqs: mod_freqs[e[4]] = 1
        else: mod_freqs[e[4]] += 1
    data['events'].progress_apply(lambda ls: [add_to_freqs(e) for e in ls])
    all_wordid_freqs = [subj_freqs, verb_freqs, obj_freqs, prep_freqs, mod_freqs]
    # drop all words lost when dropping events
    remaining_wordids = set(subj_freqs).union(set(verb_freqs)).union(set(obj_freqs)).union(set(prep_freqs)).union(set(mod_freqs))
    for w in words_ids:
        if not (words_ids[w] in remaining_wordids):
            words_ids.pop(w)

    #4. separate features
    s_all = np.stack(data['events'].map(lambda ls: [e[0] for e in ls]))
    v_all = np.stack(data['events'].map(lambda ls: [e[1] for e in ls]))
    o_all = np.stack(data['events'].map(lambda ls: [e[2] for e in ls]))
    p_all = np.stack(data['events'].map(lambda ls: [e[3] for e in ls]))
    m_all = np.stack(data['events'].map(lambda ls: [e[4] for e in ls]))
#    topic_inputs = tf.constant(np.stack(data['topic'].values))
#    genre_inputs = tf.constant(np.stack(data['genres'].values))
    
    #5. input_list = all first events
    input_list = np.array([
        np.array([ np.array([s_all[i][1], v_all[i][1], o_all[i][1], p_all[i][1], m_all[i][1]])] )
        for i in range(len(s_all))])
    
    #6. dec_inps_list = all event sequences without first events (include start bits)
    s_all = np.delete(s_all, 1, axis=1)
    v_all = np.delete(v_all, 1, axis=1)
    o_all = np.delete(o_all, 1, axis=1)
    p_all = np.delete(p_all, 1, axis=1)
    m_all = np.delete(m_all, 1, axis=1)
    dec_inps_list = np.array([
            [ np.array([s_all[sti][wi], v_all[sti][wi], o_all[sti][wi], p_all[sti][wi], m_all[sti][wi]]) 
                for wi in range(len(s_all[sti])) ] 
        for sti in range(len(s_all)) ])
    
    #7. outputs_x = event sequences for feature x without start bits or first events (plus extra padding bit at end)
    outputs_s = np.array([np.append(s_all[i][1:], 2) for i in range(len(s_all))])
    outputs_v = np.array([np.append(v_all[i][1:], 2) for i in range(len(v_all))])
    outputs_o = np.array([np.append(o_all[i][1:], 2) for i in range(len(o_all))])
    outputs_p = np.array([np.append(p_all[i][1:], 2) for i in range(len(p_all))])
    outputs_m = np.array([np.append(m_all[i][1:], 2) for i in range(len(m_all))])

    print("Finished encoding data.")

    print("input_list shape:", input_list.shape)        # should be: ({total_stories}, 1, 5)
    print("dec_inps_list shape:", dec_inps_list.shape)  # should be: ({total_stories}, {max_story_len}, 5)
    print("outputs_v shape:", outputs_v.shape)          # should be: ({total_stories}, {max_story_len})

    all_subjs = set(all_wordid_freqs[0].keys())
    all_subjs.update([1])
    all_verbs = set(all_wordid_freqs[1].keys())
    all_verbs.update([1])
    all_objs = set(all_wordid_freqs[2].keys())
    all_objs.update([0, 1])
    all_preps = set(all_wordid_freqs[3].keys())
    all_preps.update([0, 1])
    all_mods = set(all_wordid_freqs[4].keys())
    all_mods.update([0, 1])
    words_by_feature = [list(all_subjs), list(all_verbs), list(all_objs), list(all_preps), list(all_mods)]
    vocab_size = len(all_words)

    return input_list, dec_inps_list, outputs_s, outputs_v, outputs_o, outputs_p, outputs_m, vocab_size, words_by_feature, words_ids, all_wordid_freqs, genres_idxs

def check_rewards(data_folder, verb_inputs, verb_freqs, target_verbs, frm=0):
    print("Saving rewards...")
    romance_rewards = calc_rewards(verb_inputs, verb_freqs, target_verbs, frm)
    with open(os.path.join(data_folder, 'rewards\\love_marry_kiss.txt'), 'w') as f:
        f.write(json.dumps(romance_rewards))
    return romance_rewards

def main():
    genre_clusters = 0     #set to 0 to skip LDA
    data_folder = 'C:\\Users\\Amy\\Documents\\work\\uni\\II\\project_data'
    import_from = 'datasets\\transformed_datasets'
    export_to = 'models\\seq2seq'
    with open(os.path.join(data_folder, import_from, 'dataset_params.txt'), 'r') as f:
        params = json.load(f)

    # 1. prepare data for training
    plots, metadata = import_data(data_folder, import_from)
    limit = 500
    if limit < len(list(plots.index.unique())):
        plots = plots.loc[list(range(limit))]
        metadata = metadata.loc[list(range(limit))]

    input_list, dec_inps_list, outputs_s, outputs_v, outputs_o, outputs_p, outputs_m, vocab_size, words_by_feature, words_ids, all_wordid_freqs, genres_idxs = encode_train(data_folder, plots, metadata, genre_clusters, generalised=params['generalised'])
    target_verbs = [words_ids[v] for v in ['love.v.01', 'marry.v.01', 'snog.v.01']]
    freq_scaling_c = 0.1 #recommended value: 0.05
    if (freq_scaling_c < 0) or (freq_scaling_c > 0.1):
        print("Invalid value for frequency scaling constant freq_scaling_c: should be in range {0;1}  [default: 0 for no scaling]")

    # 2. define parameters for and check clustering for reward function
    clustering = {'no_of_breaks':40, 'min':10}  #{'no_of_breaks':30, 'min':20} or None
    frm = 0.01   #frequency reward modifier: recommended value = 0 or 0.01
    if (frm < 0) or (frm > 0.5):
        print("Invalid value for frequency reward multiplier frm: should be in range {0;0.5}  [default: 0 for no scaling]")

    # [debug: export rewards to explore separately]
#    romance_rewards = check_rewards(data_folder, outputs_v, all_wordid_freqs[1], target_verbs, frm)
#    _, rewards_df = cluster_verbs(romance_rewards, no_of_breaks=clustering['no_of_breaks'], min=clustering['min'])
#    groups = rewards_df.groupby('cluster').groups
#    avgs = [(g, np.mean([rewards_df.loc[w]['reward'] for w in groups[g]])) for g in groups]
#    print("Cluster means", avgs)
#    exit()

    # 3. define and train model
    print("Training model...")
    model, encoder_model, decoder_model, verb_clusters, freq_scalers, G_hist = rrun(input_list, dec_inps_list, 
                        outputs_s, outputs_v, outputs_o, outputs_p, outputs_m, 
                        target_verbs, all_wordid_freqs, words_by_feature, vocab_size,
                        clustering=clustering, frm=frm, freq_scaling_c=freq_scaling_c,
                        max_story_len=15, pretrain_epochs=32, rl_epochs=10, steps_per_execution=10000) 
                        # pretrain_epochs=128, rl_epochs=32, steps_per_execution=10000
    # 4. save model
    print("Saving model...")
    encoder_model.compile(optimizer="adam", loss="categorical_crossentropy")
    decoder_model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.save(os.path.join(data_folder, export_to, "model.model"))
    encoder_model.save(os.path.join(data_folder, export_to, "encoder_model.model"))
    decoder_model.save(os.path.join(data_folder, export_to, "decoder_model.model"))

    # export conversion dictionaries
    print("Exporting parameters and auxiliary data...")
    with open(os.path.join(data_folder, export_to, 'genres_idxs.txt'), 'w') as f:
        f.write(json.dumps(genres_idxs))
    with open(os.path.join(data_folder, export_to, 'words_ids.txt'), 'w') as f:
        f.write(json.dumps(words_ids))
    with open(os.path.join(data_folder, export_to, 'words_by_feature.txt'), 'w') as f:
        words_by_feature = [[str(w) for w in words_by_feature[i]] for i in range(5)]
        f.write(json.dumps(words_by_feature))
    # export verb clustering
    with open(os.path.join(data_folder, export_to, 'verb_clusters.txt'), 'w') as f:
        if (clustering != None):
            f.write(json.dumps(verb_clusters))
        else:
            f.write("0")
    # export frequency scaling parameters
    with open(os.path.join(data_folder, export_to, 'freq_scalers.txt'), 'w') as f:
        if freq_scaling_c != 0:
            freq_scalers = [list(scaler) for scaler in freq_scalers]
        f.write(json.dumps(freq_scalers))
    # export parameters
    if genre_clusters > 0:
        params['lda'] = True
    else:
        params['lda'] = False
    params['freq_scaling_c'] = freq_scaling_c
    with open(os.path.join(data_folder, export_to, 'train_params.txt'), 'w') as f:
        f.write(json.dumps(params))

    print("Done.")

    # 5. plot training reward history (RL)
    plt.plot(range(len(G_hist)), G_hist)
    plt.xlabel("Num of training episodes")
    plt.ylabel("$Story Score$")
    plt.show()

        
if __name__ == '__main__':
    main()
