#!/usr/bin/env python
# coding: utf-8

import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from reinforce import decode_sequence
from tqdm import tqdm
tqdm.pandas()

def preprepared_data(words_ids, params):
    #import test data
    with open(os.path.join('C:\\Users\\Amy\\Documents\\work\\uni\\II\\project_data', 'datasets', 'test_data.txt'), 'r') as f:
        test_data = json.load(f)
    first_sentences = [list.copy(story) for story in test_data]
    keep_si = []
    input_list = []
    for si in range(len(test_data)):
        reject = False
        for wi in range(5):
            w = test_data[si][wi]
            if w in words_ids:
                test_data[si][wi] = words_ids[w]
            else:
                reject = True
                print(w, "not in vocabulary - event rejected:", test_data[si])
                break
        if not reject:
            input_list.append(np.array([np.array(test_data[si])]))
            keep_si.append(si)
    input_list = np.array(input_list)
    first_sentences = [first_sentences[si] for si in keep_si]
    return first_sentences, input_list

def predict(model, encoder_model, decoder_model, input_list, v_targets, words_by_feature, vocab_size, ids_words, freq_scaling_c=0, freq_scalers=[], max_story_len=15, verb_clusters=None, c_max=0):
    results = []
    is_success = []         # 'did output succeed in reaching a target verb?'
    verb_surprisals = []    # surprisal (log probability) of each output verb 
    perplexities = []       # perplexity of each output story

    def perplexity(px):
        # entropy = sum_x( p(x) * -log(p(x)) )
        entropy = np.sum([(0 - x) * np.log2(x) for x in px])
        # perplexity = 2 ^ entropy
        return 2 ** entropy
    
    def surprisal(px):
        # surprisal = -log(p(x))
        return [0 - np.log2(x) for x in px]

    print("Generating stories for input sentences...")
    for inp_st in tqdm(input_list):
        res, _, out_verb_probs = decode_sequence(False, inp_st, encoder_model, decoder_model, words_by_feature, vocab_size, freq_scaling_c, freq_scalers, max_story_len, verb_clusters, c_max)
        res = [[ids_words[idw] for idw in event] for event in res]
        target_reached = False
        res_trimmed = []
        for ev in res:
            if target_reached == False:
                res_trimmed.append(ev)
            if (ev[1] in v_targets) or (ev[2] in v_targets):    #if verb or xcomp obj == target verb
                target_reached = True
        is_success.append(target_reached)
        results.append(res_trimmed)
        verb_surprisals.append(surprisal(out_verb_probs))
        perplexities.append(perplexity(out_verb_probs))

    hits = is_success.count(True)
    hit_rate = hits / len(input_list)
    print("Target hit rate:", hit_rate, "(" + str(hits) + "/" + str(len(input_list)) + " stories)")
    avg_len = np.mean([len(story) for story in results])
    print("Average output story length:", avg_len)
    results_successes_only = [results[si] for si in np.where(is_success)[0]]
    avg_len_hit = np.mean([len(story) for story in results_successes_only])
    print("Average output story length (hits only):", avg_len_hit)
    print("Average output perplexity:", np.mean(perplexities))
    perplexities_successes_only = [perplexities[si] for si in np.where(is_success)[0]]
    print("Average output perplexity (hits only):", np.mean(perplexities_successes_only))

    kernel_size = 3
    kernel = np.ones(kernel_size) / kernel_size

    plt.title("Smoothed Verb Surprisal Over Time for Generated Plots (All)")
    for i in range(len(res_trimmed)):
        plt.plot(np.convolve(verb_surprisals[i], kernel, mode='same'))
    plt.xlabel("Progress through Story")
    plt.ylabel("$Surprisal$")
    #plt.xlim([0, max_story_len])
    #surprisals_flat = [item for sublist in verb_surprisals for item in sublist]
    #plt.ylim([np.min(surprisals_flat), np.max(surprisals_flat)])
    plt.show()

    plt.title("Smoothed Verb Surprisal Over Time for Generated Plots (Hits Only)")
    for i in range(len(is_success)):
        if is_success[i]:
            plt.plot(np.convolve(verb_surprisals[i][:len(res_trimmed)], kernel, mode='same'))
    plt.xlabel("Progress through Story")
    plt.ylabel("$Surprisal$")
    plt.show()

    return results, is_success, verb_surprisals, perplexities

def compare_reward_journeys(plots_reward_journeys, preds_reward_journeys, c):
    #reward_journey = [rewards[words_ids[ev[1]]] for ev in story]       BUT CUT PLOTS AT 222222

    kernel_size = 10
    kernel = np.ones(kernel_size) / kernel_size
    norm_plots_reward_journeys = []
    norm_preds_reward_journeys = []

    for si in range in len(plots_reward_journeys):
        reward_journey = plots_reward_journeys[si]
        step = c / len(reward_journey)
        step_indices = [np.floor(i * step) for i in range(c)]
        reward_journey = np.convolve(reward_journey, kernel, mode='same')   #smooth rewards over time by rolling average
        reward_journey_summary = [reward_journey[i] for i in step_indices]  #sample c rewards from smoothed array
        reward_journey_summary = np.convolve(reward_journey_summary, kernel, mode='same')  #smooth by rolling average again to account for stepping
        norm_plots_reward_journeys.append(reward_journey_summary)

    for si in range in len(preds_reward_journeys):
        reward_journey = preds_reward_journeys[si]
        step = c / len(reward_journey)
        step_indices = [np.floor(i * step) for i in range(c)]
        reward_journey = np.convolve(reward_journey, kernel, mode='same')   #smooth rewards over time by rolling average
        reward_journey_summary = [reward_journey[i] for i in step_indices]  #sample c rewards from smoothed array
        reward_journey_summary = np.convolve(reward_journey_summary, kernel, mode='same')  #smooth by rolling average again to account for stepping
        norm_preds_reward_journeys.append(reward_journey_summary)

    plt.title("Reward Journeys (Corpus)")
    plt.plot(norm_plots_reward_journeys)
    plt.xlabel("Progress through Story")
    plt.ylabel("$Reward (smoothed)$")
    plt.show()

    plt.title("Reward Journeys (Generated Plots)")
    plt.plot(norm_preds_reward_journeys)
    plt.xlabel("Progress through Story")
    plt.ylabel("$Reward (smoothed)$")
    plt.show()

    plots_avg_r_by_step = np.zeros(c)
    preds_avg_r_by_step = np.zeros(c)
    for step in range(c):
        plots_avg_r_by_step[step] = np.mean([rs[step] for rs in norm_plots_reward_journeys])
        preds_avg_r_by_step[step] = np.mean([rs[step] for rs in norm_preds_reward_journeys])

    plt.title("Average Reward Journey")
    plt.plot(plots_avg_r_by_step, 'r-', label="Corpus")
    plt.plot(preds_avg_r_by_step, 'b-', label="Generated Plots")
    plt.xlabel("Progress through Story")
    plt.ylabel("$Reward (smoothed)$")
    plt.show()

def plot_survey():
    labels = ["Likability", "Total Utility", "Partial Utility", "Plausibility", "Local Causality", "Elaboration",
              "Novelty", "Non-Repetition", "Cohesiveness", "Grammaticality", "Commonsensicality", "Genre-Relevance"]
    res = [[3,2,5,6,7,7,10,10,4,7,3,1],
           [7,6,9,7,6,4,8,10,8,8,7,10],
           [3,1,1,2,1,7,10,10,2,7,1,5],
           [2,0,0,0,0,1,8,10,0,6,0,3],
           [9,7,5,6,7,5,7,10,6,4,5,8],
           [4,3,5,5,7,5,7,10,3,4,4,6]]
    
    #bar widths
    width = 0.15
    br1 = np.arange(len(res[0]))
    br2 = [x + width for x in br1]
    br3 = [x + width for x in br2]
    br4 = [x + width for x in br3]
    br5 = [x + width for x in br4]
    
    plt.barh(br1, res[0], height=width)
    plt.barh(br2, res[1], height=width)
    plt.barh(br3, res[2], height=width)
    plt.barh(br4, res[3], height=width)
    plt.barh(br5, res[4], height=width)
    plt.xlabel("Rating (0 - 10)")
    plt.ylabel("Attributes")
    plt.yticks([r + width for r in range(len(res[0]))], labels)
    plt.show()

    res_rotated = [list(r) for r in zip(*res[::-1])]
    df = pd.DataFrame(res_rotated, index=labels)
    df.T.boxplot(vert=False)
    plt.subplots_adjust(left=0.25)
    plt.xlabel("Rating (0 - 10)")
    plt.ylabel("Attributes")
    plt.show()

def main():
    data_folder = 'C:\\Users\\Amy\\Documents\\work\\uni\\II\\project_data'
    test_data_folder = 'datasets\\test_datasets'
    model_folder = 'models\\seq2seq'

    max_story_len = 15
    freq_scalers=[]
    v_targets = ['love.v.01', 'marry.v.01', 'snog.v.01']

    # import data
    with open(os.path.join(data_folder, model_folder, 'train_params.txt'), 'r') as f:
        params = json.load(f)
    freq_scaling_c = float(params['freq_scaling_c'])
    with open(os.path.join(data_folder, model_folder, 'genres_idxs.txt'), 'r') as f:
        genres_vec_idx = json.load(f)
    with open(os.path.join(data_folder, model_folder, 'words_ids.txt'), 'r') as f:
        words_ids = json.load(f)
        words_ids.pop("null")
        words_ids[None] = 3
    ids_words = {words_ids[w]: w for w in words_ids}
    vocab_size = len(words_ids)
    with open(os.path.join(data_folder, model_folder, 'words_by_feature.txt'), 'r') as f:
        words_by_feature = json.load(f)
        words_by_feature = [[int(w) for w in words_by_feature[i]] for i in range(5)]
    with open(os.path.join(data_folder, model_folder, 'verb_clusters.txt'), 'r') as f:
        verb_clusters = json.load(f)
        if type(verb_clusters) == int:
            verb_clusters = None
            c_max = 0
        else:
            c_max = max(verb_clusters)
    with open(os.path.join(data_folder, model_folder, 'freq_scalers.txt'), 'r') as f:
        freq_scalers = json.load(f)
        if freq_scaling_c != 0:
            freq_scalers = np.array([np.array(scaler) for scaler in freq_scalers])

    first_sentences, input_list = preprepared_data(words_ids, params)

    plot_survey()

    # load model
    model = keras.models.load_model(os.path.join(data_folder, model_folder, 'model.model'))
    encoder_model = keras.models.load_model(os.path.join(data_folder, model_folder, 'encoder_model.model'))
    decoder_model = keras.models.load_model(os.path.join(data_folder, model_folder, 'decoder_model.model'))
    results, is_success, verb_surprisals, perplexities = predict(model, encoder_model, decoder_model, input_list, v_targets, words_by_feature, vocab_size, ids_words, freq_scaling_c, freq_scalers, max_story_len, verb_clusters, c_max)

    #svo = [[first_sentences[i]] + [event[:3] for event in results[i]] for i in range(len(results))]
    #svom = [[first_sentences[i]] + [event[:3] + event[4] for event in results[i]] for i in range(len(results))]
    svopm = [[first_sentences[i]] + results[i] for i in range(len(results))]

    for i in range(len(svopm)):
        if is_success[i]:
            print("--------------------")
            print(svopm[i])

if __name__ == '__main__':
    main()