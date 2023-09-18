#!/usr/bin/env python
# coding: utf-8

# In[1]:
import tensorflow as tf
from keras import Input, Model
from keras import backend as K
from keras.layers import Dense, LSTM
import keras as keras
#from keras.utils import plot_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import jenkspy
from tqdm.keras import TqdmCallback
from tqdm import tqdm
tqdm.pandas()
np.seterr(all = "raise")

PAD = [2,2,2,2,2] # padding constant
GO = [1,1,1,1,1] # sets off the decoder
EOS = [2,2,2,2,2] # indicates end of sequence

# FUNCTION: calculate rewards for reinforcement learning
def calc_rewards(verb_inputs, verb_freqs, target_verbs, frm=0): 
    verbs = list(verb_freqs.keys())
    rewards = {int(v) : 0 for v in verbs}
    # 1. find distance from every verb v in each story to a given target verb
    for v_target in target_verbs:
        target_indices = [] # indices of instances of the target verb in each story
        distances = []
        for story in verb_inputs:
            story_targets = []
            for i in range(len(story)):
                if story[i] == v_target: story_targets.append(i)
            target_indices.append(story_targets)
        for story_targets in target_indices:
            inner_distances = {}
            start = 0
            for i_target in story_targets:
                for i in range(start, len(story) - i_target):
                    inner_distances[story[i]] = i_target - i
                start = i_target + 1
            inner_distances[v_target] = 0.00001 #if v = v_target
            distances.append(inner_distances)

        story_distances = {v:[] for v in verbs} #[[len(s) - dist(s, v) for v in verbs] for s in stories]
        for i_story in range(len(verb_inputs)):
            for v in verbs:
                if v in distances[i_story]: 
                    story_distances[v].append(len(verb_inputs[i_story]) - distances[i_story][v]) #len(s) - dist(s, v, v_target)
                    #note: subtracting from target rewards lower distances

        reward_distances = {}
        for v in story_distances:
            if story_distances[v] == []:
                reward_distances[v] = 0 #if verb not in story, reward is 0
            else:
                reward_distances[v] = np.log(np.sum(story_distances[v])) 

        # 2. calculate frequencies: (i) count how often each verb v appears anywhere before v_target (story_verb_freqs) 
        #                           (ii) count how often they *directly* precede v_target (action_freqs)
        story_verb_freqs = {v:0 for v in verbs}
        action_freqs = {v:0 for v in verbs}
        preceding_target_indices = [[idx - 1 for idx in story] for story in target_indices] # basically = target_indices - 1
        for i_story in range(len(target_indices)):
            if not (target_indices[i_story] == []): #if v_target in story, count everything before it
                for i_v in range(target_indices[i_story][-1]):
                    story_verb_freqs[verb_inputs[i_story][i_v]] += 1
                    if i_v in preceding_target_indices[i_story]: 
                        action_freqs[verb_inputs[i_story][i_v]] += 1
        for v in verbs:
            if (story_verb_freqs[v] == 0):
                story_verb_freqs[v] = 0.001
            if (action_freqs[v] == 0):
                action_freqs[v] = 0.001

        # 3. calculate all rewards: story_verb_freqs[v] * reward_distances[v]
        rewards = {int(v) : rewards[v] + (story_verb_freqs[v] * reward_distances[v]) for v in verbs}   

        # 3.5. multiply rewards by frequency reward multiplier [frm=0 => multiplier=1 (no effect), larger value for more shaping]
        #rewards = {v: rewards[v] * ((verb_freqs[v] ** frm)) for v in rewards}
        rewards = {v: rewards[v] * (1 + np.arcsinh(frm * verb_freqs[v])) for v in rewards}

    # 4. normalise rewards to range between -1 to 1
    all_rewards = list(rewards.values())
    rewards[PAD[1]] = np.min(all_rewards)                   #should not end in the middle
    for v_target in target_verbs:
        rewards[int(v_target)] = 1.05 * np.max(all_rewards)  #target verbs should have highest rewards
    all_rewards = list(rewards.values())
    rewards_2d = np.array(all_rewards).reshape(-1, 1)
    scaler = MinMaxScaler((-1, 1))
    scaled = scaler.fit_transform(rewards_2d)
    vs = list(rewards.keys())
    rewards = {vs[i]: scaled[i][0] for i in range(len(vs))}

    if np.isnan(list(rewards.values())).any():
        print("Error: Rewards contains N/A values.")
        exit()

    return rewards

# FUNCTION:
#   - cluster verbs based on reward using Jenks Natural Breaks optimisation
#   - for v_in in c, vocabulary of v_out is restricted to verbs in cluster c+1
#     [rest of event sampled from full distribution]
# OPTIONS:
#   - no_of_breaks: this hyperparameter should be optimised wrt 'goodness of variance fit'
def cluster_verbs(rewards_dict, no_of_breaks=150, min=None):
    rewards_df = pd.DataFrame(rewards_dict.items(), columns=['verb','reward'])
    jnb = jenkspy.JenksNaturalBreaks(no_of_breaks)
    try:
        jnb.fit(rewards_df['reward'])
    except:
        print("Rewards values too similar. Clustering failed.")
        exit()
    rewards_df['cluster'] = jnb.labels_
    if min != None:
        rewards_df = merge_clusters(rewards_df, min)
    groups = rewards_df.groupby('cluster').groups
    print("Cluster distribution:", [(g, len(groups[g])) for g in range(len(groups))])
    verb_clusters = dict(zip(rewards_df['verb'], rewards_df['cluster']))
    return verb_clusters, rewards_df

def merge_clusters(rewards_df, min=100, verbose=False):
    groups = rewards_df.groupby('cluster').groups
    group_counts = {g: len(groups[g]) for g in range(len(groups))}
    new_groups = {}
    tail = []
    tail_count = 0
    group_counter = 0
    for g in group_counts:
        if group_counts[g] > min:
            tail.append(g)
            new_groups[group_counter] = tail
            group_counter += 1
            tail = []
            tail_count = 0
        elif group_counts[g] + tail_count > min:
            tail.append(g)
            new_groups[group_counter] = tail
            group_counter += 1
            tail = []
            tail_count = 0
        else:
            tail.append(g)
            tail_count += group_counts[g]
    if len(tail) > 0:
        new_groups[group_counter] = tail
    new_group_map = {}
    for cluster in new_groups:
        for cluster_old in new_groups[cluster]:
            new_group_map[cluster_old] = cluster
    rewards_df['cluster'] = rewards_df['cluster'].map(new_group_map)
    #show new distribution:
    groups = rewards_df.groupby('cluster').groups
    if verbose:
        print("New cluster counts:", [(g, len(groups[g])) for g in range(len(groups))])
    return rewards_df

def goodness_of_variance_fit(df):
    groups = df.groupby('cluster').groups
    groups = {i: df.iloc[groups[i]]['reward'] for i in groups.keys()}
    array_mean = np.mean(df['reward'])
    ssdam = np.sum((df['reward'] - array_mean) ** 2)        # sum of squared deviations from array mean
    class_means = {g : np.mean(groups[g]) for g in groups.keys()}
    df['class_mean'] = df['cluster'].map(class_means)
    ssdcm = np.sum((df['reward'] - df['class_mean']) ** 2)  # sum of squared deviations from class means
    gvf = (ssdam - ssdcm) / ssdam
    return gvf 
    
def search_no_of_breaks(rewards_dict):
    search_space = [5, 10, 20, 40, 80, 120, 160, 180, 200] 
    best_gvf = 0
    best_n = 0
    for n in search_space:
        _, rewards_df = cluster_verbs(rewards_dict, n)
        gvf = goodness_of_variance_fit(rewards_df)
        print("GVF for ", n, " clusters: ", gvf)
        if gvf > best_gvf: 
            best_gvf = gvf
            best_n = n
    print("Best number of clusters: ", best_n, "(GVF = ", best_gvf, ")")

# FUNCTION: define and compile models
def create_model(vocab_size, steps_per_execution):
#    def custom_loss(y_true, y_pred):
#        out = K.clip(y_pred, 1e-8, 1-1e-8)
#        log_lik = y_true*K.log(out)
#        return K.sum(log_lik * advantages)
        
    num_encoder_tokens = 5
    num_decoder_tokens = 5
    latent_dim = 256
    
    # these need to be set to your various vocab sizes
    num_s_tokens = vocab_size
    num_v_tokens = vocab_size
    num_o_tokens = vocab_size
    num_p_tokens = vocab_size
    num_m_tokens = vocab_size

    encoder_inputs= Input(shape=(None, num_encoder_tokens), name="enc_input")
    encoder_lstm = LSTM(latent_dim, return_state=True) 
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]     #discard encoder_outputs and only keep the states

    # set up the decoder, using encoder_states as initial state
    decoder_inputs= Input(shape=(None, num_decoder_tokens), name="dec_input")
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _,_ = decoder_lstm(decoder_inputs, initial_state=encoder_states)    
    
    decoder_dense_s = Dense(num_s_tokens, activation='softmax')
    decoder_dense_v = Dense(num_v_tokens, activation='softmax')
    decoder_dense_o = Dense(num_o_tokens, activation='softmax')
    decoder_dense_p = Dense(num_p_tokens, activation='softmax')
    decoder_dense_m = Dense(num_m_tokens, activation='softmax')

    output_s = decoder_dense_s(decoder_outputs)
    output_v = decoder_dense_v(decoder_outputs)
    output_o = decoder_dense_o(decoder_outputs)
    output_p = decoder_dense_p(decoder_outputs)
    output_m = decoder_dense_m(decoder_outputs)
    
    # define the model that will turn encoder_input_data and decoder_input_data into decoder_target_data
    model = Model(inputs=[encoder_inputs]+[decoder_inputs], outputs=[output_s, output_v, output_o, output_p, output_m], name="full_model")
    # plot the model
    print(model.summary())   
    # summarize model: 
#     plot_model(model, to_file='model.png', show_shapes=True)

    # define encoder inference model
    encoder_model = Model(encoder_inputs, encoder_states, name="encoder")
    print(encoder_model.summary())

    # define decoder inference model
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, d_state_h, d_state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [d_state_h, d_state_c]
    
    decoder_dense_s = Dense(num_s_tokens, activation='softmax')
    decoder_dense_v = Dense(num_v_tokens, activation='softmax')
    decoder_dense_o = Dense(num_o_tokens, activation='softmax')
    decoder_dense_p = Dense(num_p_tokens, activation='softmax')
    decoder_dense_m = Dense(num_m_tokens, activation='softmax')

    output_s = decoder_dense_s(decoder_outputs)
    output_v = decoder_dense_v(decoder_outputs)
    output_o = decoder_dense_o(decoder_outputs)
    output_p = decoder_dense_p(decoder_outputs)
    output_m = decoder_dense_m(decoder_outputs)
    
    decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[output_s, output_v, output_o, output_p, output_m] + decoder_states, name="decoder")
    print(decoder_model.summary())
    
    # summarize model:
#     plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True)
#     plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", steps_per_execution=steps_per_execution)
    
    return model, encoder_model, decoder_model
    
def decode_sequence(is_train_step, input_seq, encoder_model, decoder_model, words_by_feature, vocab_size, freq_scaling_c=0, freq_scalers=[], max_out_l=15, verb_clusters=None, c_max=0):
    target_seq = [[1,1,1,1,1],]
    decoded_sentence = []
    act_probs = []
    out_verb_probs = []
    input_seq = np.array(input_seq)
    input_seq = input_seq.reshape(1,-1,5)
    
    end= False
    ix=1
    s_index = -1
    v_index = -1 
    o_index = -1 
    m_index = -1
    p_index = -1
    s_choices = list(range(vocab_size)) #words_by_feature[0]
    v_choices = list(range(vocab_size)) #words_by_feature[1]
    o_choices = list(range(vocab_size)) #words_by_feature[2]
    p_choices = list(range(vocab_size)) #words_by_feature[3]
    m_choices = list(range(vocab_size)) #words_by_feature[4]
   
    states_value = encoder_model.predict(input_seq, verbose=0, workers=16)  
    while True:        
        t_seq = target_seq
        t_seq = np.reshape(t_seq,(1,-1,5))
        os, ov, oo, op, om, h, c = decoder_model.predict([t_seq] + states_value, verbose=0, workers=16)
        
        nx_s_dist = os[0, -1]
        nx_v_dist = ov[0, -1]
        nx_o_dist = oo[0, -1]
        nx_p_dist = op[0, -1]
        nx_m_dist = om[0, -1]
        act_probs.append(list(nx_v_dist.ravel()))

        # limit output vocab for prediction
        v_current = t_seq[-1][-1][1]
        if (verb_clusters != None) and (v_current != GO[1]) and (v_current in verb_clusters):
            # limit output verbs to HIGHER numbered (higher prob, closer to target) clusters only: c_next > c_current
            c_current = verb_clusters[v_current]
            v_selection = []
            for v_next, c_next in verb_clusters.items():
                if (c_next == c_max and c_current == c_max) or (c_next == c_current + 1):   #if next verb is in next class, or already max class
                    v_selection.append(v_next)
            mask = np.ones(vocab_size, dtype=bool)
            mask[v_selection] = False
            nx_v_dist[mask] = 0
        else:
            mask = np.ones(vocab_size, dtype=bool)
            mask[words_by_feature[1]] = False  #limit vocab to verbs only
            nx_v_dist[mask] = 0
        mask = np.ones(vocab_size, dtype=bool)
        mask[words_by_feature[0]] = False
        nx_s_dist[mask] = 0
        mask = np.ones(vocab_size, dtype=bool)
        mask[words_by_feature[2]] = False
        nx_o_dist[mask] = 0
        mask = np.ones(vocab_size, dtype=bool)
        mask[words_by_feature[3]] = False
        nx_p_dist[mask] = 0
        mask = np.ones(vocab_size, dtype=bool)
        mask[words_by_feature[4]] = False
        nx_m_dist[mask] = 0
        # scale probalities by word frequency (how common a word is)
        if (freq_scaling_c > 0):
            nx_s_dist = np.array([nx_s_dist[i] * freq_scalers[0][i] for i in range(vocab_size)])
            nx_v_dist = np.array([nx_v_dist[i] * freq_scalers[1][i] for i in range(vocab_size)])
            nx_o_dist = np.array([nx_o_dist[i] * freq_scalers[2][i] for i in range(vocab_size)])
            nx_p_dist = np.array([nx_p_dist[i] * freq_scalers[3][i] for i in range(vocab_size)])
            nx_m_dist = np.array([nx_m_dist[i] * freq_scalers[4][i] for i in range(vocab_size)])
        nx_s_dist = nx_s_dist/nx_s_dist.sum() #normalise probabilities to add to 1 again
        nx_v_dist = nx_v_dist/nx_v_dist.sum()
        nx_o_dist = nx_o_dist/nx_o_dist.sum()
        nx_p_dist = nx_p_dist/nx_p_dist.sum()
        nx_m_dist = nx_m_dist/nx_m_dist.sum()
         
        s_index = np.random.choice(s_choices, p=nx_s_dist)
        v_index = np.random.choice(v_choices, p=nx_v_dist)
        o_index = np.random.choice(o_choices, p=nx_o_dist)
        p_index = np.random.choice(p_choices, p=nx_p_dist)
        m_index = np.random.choice(m_choices, p=nx_m_dist)

        if (not is_train_step):
            pv = nx_v_dist[v_choices.index(v_index)]
            out_verb_probs.append(pv)

        decoded_sentence += [[int(s_index), int(v_index), int(o_index), int(p_index), int(m_index)]] #sampled_char
        if len(decoded_sentence)==max_out_l or 2==int(v_index):
            break
        target_seq = [[int(s_index), int(v_index), int(o_index), int(p_index), int(m_index)], ]
        states_value = [h, c]
           
    K.clear_session()
    return decoded_sentence, act_probs, out_verb_probs

def update_policy(inp_tok, model, story, act_probs, G, G_hist):
    ALPHA=0.01
    GAMMA=0.9

    inp_tok = np.reshape(inp_tok,(1,-1,5))
    dec_inputs = [GO] + story[:-1]

    G_hist_flat = [item for sublist in G_hist for item in sublist]
    avg_r = np.mean(G_hist_flat)
    std_r = np.std(G_hist_flat)
    
    rewards = G
#     discounted_rewards=[]
#     cumulative_total_return=0
#     for reward in rewards[::-1]:
#         cumulative_total_return = (GAMMA*cumulative_total_return)+reward
#         discounted_rewards.insert(0, cumulative_total_return)
#     discounted_rewards = rewards
#     advantage = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards)+1e-6)
    try:
        advantage = (rewards - avg_r) / (std_r)
        advantage = np.reshape(advantage,(1, -1, 1))
        advantage = ALPHA * advantage
    except:
        print("An error occured while calculating advantage.")
        print(rewards)
        print(avg_r)
        print(std_r)
        exit()

    subjs = np.array([event[0] for event in story]).reshape((1, len(story), 1))
    verbs = np.array([event[1] for event in story]).reshape((1, len(story), 1))
    objs = np.array([event[2] for event in story]).reshape((1, len(story), 1))
    preps = np.array([event[3] for event in story]).reshape((1, len(story), 1))
    mods = np.array([event[4] for event in story]).reshape((1, len(story), 1))
    
    dec_inputs = np.reshape(dec_inputs, (1,-1,5))

    if not np.any(np.isinf(advantage)):
        model.fit([inp_tok, dec_inputs], [subjs, verbs, objs, mods, preps],  epochs=1, sample_weight=advantage, verbose=0, workers=16)
    
    else:
        print("Infinite advantages detected - check for convergence!")
    return model

def rrun(input_list, dec_inps_list, outputs_s, outputs_v, outputs_o, outputs_p, outputs_m, target_verbs, wordid_freqs, words_by_feature, vocab_size, clustering=None, frm=0, freq_scaling_c=0, max_story_len=15, pretrain_epochs=64, rl_epochs=64, steps_per_execution=10000):
    max_train_len = len(outputs_s[0])
    pretrain  = True
    reinforce = True

    print("Calculating rewards...")
    rewards = calc_rewards(outputs_v, wordid_freqs[1], target_verbs, frm)
    verb_clusters = None
    c_max = 0
    if clustering != None:
        print("Clustering rewards...")
        verb_clusters, rewards_df = cluster_verbs(rewards, no_of_breaks=clustering['no_of_breaks'], min=clustering['min'])
        groups = rewards_df.groupby('cluster').groups
        c_max = max(list(groups))

    freq_scalers = []
    if freq_scaling_c > 0:
        s_freq_scaler = np.ones(vocab_size, dtype=float)
        for w in wordid_freqs[0]:
            s_freq_scaler[w] = 1 + np.arcsinh(freq_scaling_c * wordid_freqs[0][w])
        v_freq_scaler = np.ones(vocab_size, dtype=float)
        for w in wordid_freqs[1]:
            v_freq_scaler[w] = 1 + np.arcsinh(freq_scaling_c * wordid_freqs[1][w])
        o_freq_scaler = np.ones(vocab_size, dtype=float)
        for w in wordid_freqs[2]:
            o_freq_scaler[w] = 1 + np.arcsinh(freq_scaling_c * wordid_freqs[2][w])
        p_freq_scaler = np.ones(vocab_size, dtype=float)
        for w in wordid_freqs[3]:
            p_freq_scaler[w] = 1 + np.arcsinh(freq_scaling_c * wordid_freqs[3][w])
        m_freq_scaler = np.ones(vocab_size, dtype=float)
        for w in wordid_freqs[4]:
            m_freq_scaler[w] = 1 + np.arcsinh(freq_scaling_c * wordid_freqs[4][w])
        freq_scalers = [s_freq_scaler, v_freq_scaler, o_freq_scaler, p_freq_scaler, m_freq_scaler]

    model, encoder_model, decoder_model = create_model(vocab_size, steps_per_execution)
    if pretrain:
        print("Pretraining...")
        model.fit([input_list, dec_inps_list], [outputs_s, outputs_v, outputs_o, outputs_p, outputs_m], epochs=pretrain_epochs, verbose=0, callbacks=[TqdmCallback(verbose=1)], workers=16)

    G_hist=[]
    G_hist_avg = []

    if reinforce:
        outputs_all = np.zeros((len(outputs_s), 5, max_train_len))
        for i in range(len(outputs_s)):
            outputs_all[i] = np.array([outputs_s[i], outputs_v[i], outputs_o[i], outputs_p[i], outputs_m[i]])
        print("Doing reinforcement learning...")
        for k in tqdm(range(rl_epochs)):
            for inp_st in tqdm(input_list):
                G=0
                output_story, act_probs, _ = decode_sequence(True, inp_st, encoder_model, decoder_model, words_by_feature, vocab_size, freq_scaling_c, freq_scalers, max_story_len, verb_clusters, c_max)
                G = [rewards[ev[1]] for ev in output_story]
                G_hist.append(G)
                model = update_policy(inp_st, model, output_story, act_probs, G, G_hist)
            G_hist_avg.append(np.mean([item for sublist in G_hist for item in sublist]))
            
    return model, encoder_model, decoder_model, verb_clusters, freq_scalers, G_hist_avg
