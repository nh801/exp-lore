#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import json
from json import JSONDecodeError
import re
import nltk
from nltk.corpus import wordnet as wn
from stanfordcorenlp import StanfordCoreNLP
from collections import Counter, OrderedDict
from tqdm import tqdm
tqdm.pandas()

data_folder = 'C:\\Users\\Amy\\Documents\\work\\uni\\II\\project_data'
stanford_folder = 'C:\\Users\\Amy\\Documents\\work\\uni\II\\project_data\\models\\stanford-corenlp-4.5.1'
import_from = 'datasets\\cmu_movies'
export_to = 'datasets\\transformed_datasets'

nlp = StanfordCoreNLP(stanford_folder, memory='8g') #quiet=False, logging_level=logging.DEBUG
port = nlp.port

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', 500)

def clean_genres(plots, metadata, romance_only=True, reduced=False, output_check=False):
    print("Cleaning genre data...")
    genre_folder = os.path.join(data_folder, 'genre_cleaning')
    metadata['genres'] = metadata['genres'].str.lower()
    metadata['genres'] = metadata['genres'].map(lambda d: set(json.loads(d).values()))
    
    #1. delete_tagged (drop rows with any item from {set} in genres)
    delete_tagged = set(line.strip() for line in open(os.path.join(genre_folder, 'delete_tagged.txt')))
    drop_rows = metadata[metadata['genres'].map(lambda s: len(s.intersection(delete_tagged)) > 0)].index
    metadata.drop(drop_rows, inplace=True)
    plots.set_index('wiki_id', inplace=True)
    plots.drop(drop_rows, inplace=True, errors='ignore')     #equally drop from plots
    plots.reset_index(inplace=True)
    #2. combine, rename and supplement 
        # 2.1. regex name changes
    replaces = [' story', ' films', ' film', ' movies', ' movie', ' cinema', ' fiction']
    def clean_suffix(g):
        for x in replaces:
            g = g.replace(x, '')
        return g
    metadata['genres'] = metadata['genres'].map(lambda s: set(clean_suffix(g) for g in s))
        #2.2. if has key, add values to set; then remove keys
    combine_rename = dict([[gs.strip() for gs in line.split("\t")] for line in open(os.path.join(genre_folder, 'combine_and_rename.txt'))])
    combine_rename.update({g: combine_rename[g].split(',') for g in combine_rename.keys()})
    metadata['genres'].map(lambda s: [s.update(combine_rename[g]) for g in s.intersection(combine_rename)])
    metadata['genres'] = metadata['genres'].map(lambda s: s - set(combine_rename.keys()))
        #2.3. supplement
    supplement = dict([[gs.strip() for gs in line.split("\t")] for line in open(os.path.join(genre_folder, 'supplement.txt'))])
    supplement.update({g: supplement[g].split(',') for g in supplement.keys()})
    metadata['genres'].map(lambda s: [s.update(supplement[g]) for g in s.intersection(supplement)])
    metadata['genres'].map(lambda s: [s.update(supplement[g]) for g in s.intersection(supplement)]) #twice in case category added
    #3. remove unwanted tags
        #3.1. get genres with <35 entries to remove from all sets
    genre_counts = pd.Series([x for item in metadata['genres'] for x in item]).value_counts() #count works in each genre
    low_counts = set()
    if (not reduced):
        low_counts = set(genre_counts.index[genre_counts < 35].tolist())
        #3.2. remove any genres in delete_tags from all sets
    delete_tags = set(line.strip() for line in open(os.path.join(genre_folder, 'delete_tags.txt')))
    delete_tags.update(low_counts)
    delete_tags.add('') #remove any blank genres
    metadata['genres'] = metadata['genres'].map(lambda s: s - delete_tags)
    # [4. ROMANCE ONLY]
    if (romance_only):
        not_romance = metadata[metadata['genres'].map(lambda s: not('romance' in s))].index
        metadata.drop(not_romance, inplace=True)
        plots.set_index('wiki_id', inplace=True)
        plots.drop(not_romance, inplace=True, errors='ignore')     #equally drop from plots
        plots.reset_index(inplace=True)
    # 4. check (output genre counts to file)
    if output_check:
        genre_counts = pd.Series([x for item in metadata['genres'] for x in item]).value_counts()
        genre_counts.to_csv(os.path.join(genre_folder, 'genre_counts.txt'), index=True)
    metadata['genres'] = metadata['genres'].map(lambda s: list(s)) #cast to list for easier re-import later
    return plots, metadata

# OPTIONS:
#   stop_at_n_plots: limits number of plots to process (by filtering the dataframes by unique wiki_id)
def import_data(romance_only=True, stop_at_n_plots=-1):
    #1. import plot summaries
    print("Importing summaries...")
    plots = pd.read_csv(os.path.join(data_folder, import_from, "plot_summaries.txt"), header=None, names=['wiki_id', 'summary'], dtype={'wiki_id':str}, sep='\t', engine='python', quoting=3, encoding='utf-8')
    plots['summary'] = plots['summary'].map(lambda x: re.sub("The (film|episode)", "It", x))
    plots['summary'] = plots['summary'].map(lambda x: re.sub("the (film|episode)", "it", x))
    plots['summary'] = plots['summary'].map(lambda x: re.sub("Episode", "Part", x))
    plots['summary'] = plots['summary'].map(lambda x: re.sub("episode", "part", x))
    #2. import and clean metadata (genres), and prune plots
    print("Importing metadata (genres)...")
    colnames = ['wiki_id', 'freebase_id', 'title', 'release', 'revenue', 'runtime', 'lang', 'countries', 'genres']
    metadata = pd.read_csv(os.path.join(data_folder, import_from, "movie.metadata.tsv"), header=None, names=colnames, dtype={'wiki_id':str}, sep='\t', encoding='utf-8')
    metadata.set_index('wiki_id', inplace=True)
    metadata.drop(columns=['freebase_id', 'title', 'release', 'revenue', 'runtime', 'lang', 'countries'], inplace=True)

    if (stop_at_n_plots > -1):
        plots = plots.head(stop_at_n_plots)
        # drop any entries in metadata not in plots
        keep_wids = plots['wiki_id'].unique()
        all_mwids = metadata.index.unique()
        drop_mwids = list(set(all_mwids) - set(keep_wids))
        metadata.drop(drop_mwids, inplace=True, errors='ignore')
        # clean genres
        plots, metadata = clean_genres(plots, metadata, romance_only, reduced=True, output_check=True)
    else:
        # drop any entries in metadata not in plots
        keep_wids = plots['wiki_id'].unique()
        all_mwids = metadata.index.unique()
        drop_mwids = list(set(all_mwids) - set(keep_wids))
        metadata.drop(drop_mwids, inplace=True, errors='ignore')
        # clean genres
        plots, metadata = clean_genres(plots, metadata, romance_only, output_check=True)

    plots.set_index('wiki_id', inplace=True)
    return plots, metadata

# OPTIONS:
#   dep_type = "basicDependencies", "enhancedDependencies", OR "enhancedPlusPlusDependencies"
def parse_doc_with_eppd(text_doc, dep_type):
    response = nlp._request("tokenize,pos,lemma,depparse", data=text_doc)
    sents = response['sentences']
    s_out = []
    all_tokens = []
    for s in sents:
        toks = s['tokens']
        tokens_words = []
        out = {}
        eppd = s[dep_type]
        out[0] = {'tag':'TOP', 'address':0, 'deps':{}, 'lemma':None, 'rel':None, 'head':-1}
        for t in toks:
            w_ix = t['index']
            w_dict = {'address':t['index'], 'lemma':t['lemma'], 'tag':t['pos'], 'deps':{}}
            out[w_ix] = w_dict
            tokens_words.append(t['word'])
        for d in eppd:
            w_ix = d['dependent']
            rel = d['dep']
            out[w_ix]['rel'] = rel
            g_ix = d['governor']
            out[w_ix]['head'] = g_ix
            if rel not in out[g_ix]['deps']:
                out[g_ix]['deps'][rel] = []
            out[g_ix]['deps'][rel].append(w_ix)  
        s_out.append(out)
        all_tokens.append(tokens_words)
    return s_out, all_tokens

# FUNCTION:
#   parse dependencies with Stanford’s CoreNLP parser
def dependency_parse(plots, stop_at_n_plots, dep_type="basicDependencies"):
    print("Parsing dependencies...")
    generated_dependencies = []
    all_summaries_tokens = []
    if True:            
        i = 0
        for wid in tqdm(plots.index.get_level_values(0).unique().values):
            if (stop_at_n_plots > -1) and (i >= stop_at_n_plots):
                break
            i += 1 
            dep_batch, tokens = parse_doc_with_eppd(plots.loc[wid]['summary'], dep_type=dep_type)
            all_summaries_tokens.append(tokens)
            generated_dependencies.append(dep_batch)

        # keep summaries in tokenised form
        #plots.drop('summary', axis=1, inplace=True)    #DEBUG
        plots['tokens'] = all_summaries_tokens
        plots['line'] = plots['tokens'].map(lambda lst: [n for n in range(len(lst))])
        plots = plots.explode(['tokens', 'line'])
        plots.reset_index(inplace=True) #flatten
        plots.set_index(['wiki_id', 'line'], inplace=True)

        print("Exporting dependencies...")
        dependencies = []
        i = 0
        with tqdm(total=len(generated_dependencies)) as pbar:
            for deps in generated_dependencies:
                i += 1
                for dep in deps: #for each story
                    out_ls = [val for key, val in sorted(dep.items())]
                    dep_df = pd.DataFrame(out_ls)[['lemma', 'rel', 'tag', 'head', 'deps']]
                    dependencies.append(dep_df)
                pbar.update(1)
        plots['dependencies'] = dependencies

    return plots

# FUNCTION:
#   - extracts the core semantic information from each raw plot (NER)
#     -->   tries to resolve synonyms and count up certain entity types (incrementing types) 
#           so PERSON becomes PERSON_0, PERSON_1 etc.
#   - it is possible to later integrate NER results back into the plots object (see: in eventify())
# OUTPUTS:
#   entmk_list: 'entity mask' to be integrated in eventify()
#   metadata['entities']: dictionary of named objects for each plot
def create_entity_mask(plots, stop_at_n_plots, numbered_entities=False, verbose=False):
    print("Creating entity mask...")
    #parse_corefs = False      # coreference support not currently released?
    entmk_list = []
    entities_per_plot = {}
    failed_wids = []    #note: wiki_id for training data should work same as plot_id for test data
    i = 0
    for wid in tqdm(plots.index.get_level_values(0).unique().values):
        if (stop_at_n_plots > -1) and (i >= stop_at_n_plots):
            break
        i+=1
        skip = False
        if True:
            dff = plots.loc[wid]
            lines = dff['tokens'] #[i:next_i]
            full_plot = str(dff['summary'][0])

            try:
                ne_full = nlp.ner(full_plot)
                #if parse_corefs:
                #    print("getting corefs")
                #    coref_full = json.loads(nlp.annotate(full_plot, properties={'annotators':'coref', 'pipelineLanguage': 'en'}))
            except JSONDecodeError:
                failed_wids.append(wid)
                skip = True
            if not skip:
                ne_lines = []
                for line in lines:
                    ne_line, ne_full = ne_full[:len(line)], ne_full[len(line):]
                    ne_lines.append(ne_line)

                # check if NER tokenisation is different from dependency parse tokenisation
                ne_tokens = sum([len(line) for line in ne_lines])
                lines_tokens = sum([len(line) for line in lines])
                if ne_tokens != lines_tokens:
                    print("\nError: NER returned extra tokens!")
                    print("no. of tokens before NER: ", lines_tokens)
                    print("no. of tokens processed with NER: ", ne_tokens, "\n")
                    print(ne_lines)
                    exit()
                if verbose:
                    print(wid)
                    print(ne_lines)
                #pd.DataFrame(data=ne_lines).to_csv('entities.csv', index=True)
                
                sent_NEs = False
                wmap = OrderedDict()
                typ_ct = Counter()
                entity_mask = [["" for w in l] for l in ne_lines]
                if numbered_entities:
                    incrementing_types = ['PERSON', 'LOCATION', 'CITY', 'ORGANIZATION']
                else:
                    incrementing_types = []
                synonym_map = {}
                for sent_ix, ne_line in enumerate(ne_lines):
                    buildup = []
                    last_typ = "O"
                    if sent_NEs: # In this mode, reset the counter with each sentence
                        typ_ct = Counter()
                    for w_ix,ne_pr in enumerate(ne_line):
                        w,typ = ne_pr
                        if typ != last_typ: # if we've just exited a named entity
                            if buildup:
                                subterm_list = [b[0] for b in buildup]
                                ww = " ".join(subterm_list)
                                try: # first just check if there's a key match
                                    numd_typ = wmap[ww]
                                except:
                                    # this part deals with strings that have not been seen before
                                    # not increment types can be cached as-is
                                    if last_typ not in incrementing_types:
                                        numd_typ = last_typ
                                        wmap[ww] = numd_typ
                                    else: #incrementing types need a synonym check
                                        synonym = None
                                        for subterm in subterm_list:
                                            for k,v in wmap.items():
                                                if subterm == k or subterm in k.split(" ") and last_typ == v.split("_")[0]:
                                                    synonym = k
                                                    synonym_map[ww] = synonym
                                                    break
                                        if synonym:
                                            numd_typ = wmap[synonym]
                                        else: # if no synonym, assume entity is new and increment its ID
                                            numd_typ = last_typ + "_" + str(typ_ct[last_typ])
                                            typ_ct[last_typ] += 1
                                            wmap[ww] = numd_typ
                                for tok_ix in range((w_ix - len(buildup)), w_ix):
                                    entity_mask[sent_ix][tok_ix] = numd_typ 
                                #reset the entity sequence
                                buildup = []        
                        #ignore tokens outside ("O") entities, otherwise build up a sequence of tokens inside the entity
                        if typ!="O" :
                            buildup.append((w,typ))
                        last_typ = typ

                if verbose:
                    print("Synonyms, wmap, entity mask")
                    print(synonym_map)
                    print(wmap)
                    print(entity_mask)

                entities_per_plot[wid] = dict(wmap) #[(k, v) for k, v in wmap.items()]
        entmk_list.extend(entity_mask)

    entities_per_plot = pd.Series(entities_per_plot)
    entities_per_plot.index.name = 'wiki_id'
    return entmk_list, entities_per_plot.rename('entities'), failed_wids

# FUNCTION:
#   extract the core semantic information from each sentence
# OPTIONS:
#   min_events: specify a minimum number of events per story
#   generalise: generalise named entities
def eventify(plots, min_events=10, ent_mask=None):
    if ent_mask:
        print("Applying entity mask...")
        for ix in tqdm(range(len(plots))):
            dep_row = plots['dependencies'].iloc[ix]
            ent_row = ent_mask[ix]
            gen_lemmas = ["<NE>" + ent_row[j-1] if (ent_row[j-1] != "") else str(dep_row.iloc[j]['lemma']) for j in range(len(dep_row))]
            dep_row['lemma_gen'] = gen_lemmas
            plots['dependencies'].iloc[ix] = dep_row

    # 2. event translation: extract (potentially multiple) events from each sentence
    #    => event = <subject/s, verb/v, direct object/o, preposition/p, modifier noun/m>
    re_subjects = r'(nsubj(?!:))|(nsubj(:pass)?(:xsubj)?)|(xsubj)|(csubj(:pass)?)'
    re_subj_csubj = r'(nsubj(?!:))|(nsubj(:pass)?(:xsubj)?)|(xsubj)|(i?obj(?![a-z]))'
    re_objects = r'((?<!i)obj(?![a-z]))|(xcomp)'
    re_mods_obj = r"(obl:(?!(npmod|tmod|[&@]))([a-z0-9_\.\-’']+)?)|(i?obj(?![a-z]))|(nmod(?!(:[&@]))(:[a-z0-9_\.\-’']+)?)|(obl(?!(:(npmod|[&@])))(:tmod)?)|(dep)"
    re_mods_verb = r"(obl(?!(:(npmod|tmod|[&@])))(:[a-z0-9_\.\-’']+)?)|(iobj)|(nmod(?!(:[&@]))(:[a-z0-9_\.\-’']+)?)|(dep)"

    pos_verbs = set(['VB', 'VBD', 'VBZ', 'VBP', 'VBN', 'VBG'])
    all_events = []
    all_gen_events = []
    events_count = []
    pd.options.mode.chained_assignment = None   #suppress SettingWithCopy warning (as intentionally setting on copy)
    print("Eventifying...")
    with tqdm(total=len(plots['dependencies'])) as pbar:
        #for d in range(len(plots['dependencies'])):
        for dep_row in plots['dependencies']:
            #dep_row = plots['dependencies'].iloc[d]     # dep_row = df['lemma', 'rel', 'head', 'deps'] for one sentence
            subjects = []
            verbs = []
            verbs_poss = []
            objects = []
            modifiers = []
            prepositions = []
            #2.1. get possible verbs
            for i in range(1, len(dep_row)):
                if (dep_row.iloc[i]['tag'] in pos_verbs and dep_row.iloc[i]['rel'] != 'aux'):
                    verbs_poss.append(i)
            for v in verbs_poss:
                v_deps = dep_row.iloc[v]['deps']
                #2.2. get (first) subject of verb
                s = re.search(re_subjects, ' '.join(list(v_deps.keys())))
                if (s is not None):             #continue only if there is a subject
                    s_type = s.group()
                    if (s_type == 'csubj'):
                        s_deps = dep_row.iloc[v_deps.get(s_type)[0]]['deps']
                        s = re.search(re_subj_csubj, ' '.join(list(s_deps.keys())))
                        if (s is not None):
                            s = s_deps.get(s.group())[0]
                    elif (s_type == 'csubj:pass'):
                        s_deps = dep_row.iloc[v_deps.get(s_type)[0]]['deps']
                        s = re.search(re_subj_csubj, ' '.join(list(s_deps.keys())))
                        if (s_deps.get('obj') is not None):
                            s = s_deps.get('obj')[0]
                        elif (s is not None):
                            s = s_deps.get(s.group())[0]
                    else:
                        s = v_deps.get(s_type)[0]
                if (s is not None):
                    subjects.append(s)
                    verbs.append(v)
                    #2.3. get object of verb (optional)
                    o = re.search(re_objects, ' '.join(list(v_deps.keys())))
                    if (o is not None):
                        grp = o.group()
                        o = v_deps.get(grp)[0]
                        if (ent_mask is not None):  #mark as verb or object for later generalisation
                            if (dep_row.iloc[o]['lemma'][:4] != "<NE>") and (grp == "xcomp"):
                                #unnecessary alt without warning: plots.iloc[d, plots.columns.get_loc('dependencies')]['lemma_gen'][o] = "<VB>" + dep_row.iloc[o]['lemma_gen']
                                dep_row['lemma_gen'][o] = "<VB>" + dep_row.iloc[o]['lemma_gen']
                    objects.append(o)
                    #2.4. get modifier noun [of verb or object] (optional)
                    m = None
                    #2.4.1. first check if there is an object, and if modifier attached
                    if (o is not None):
                        o_deps = dep_row.iloc[o]['deps']
                        m = re.search(re_mods_obj, ' '.join(list(o_deps.keys())))
                        if (m is not None):
                            m = o_deps.get(m.group())[0]
                    #2.4.2. then check if modifier is attached to verb
                    if (m is None):
                        m = re.search(re_mods_verb, ' '.join(list(v_deps.keys())))
                        if (m is not None):
                            m = v_deps.get(m.group())[0]
                    modifiers.append(m)
                    #2.5. get preposition [of modifier]
                    if (m is not None):
                        m_deps = dep_row.iloc[m]['deps']
                        p = m_deps.get('case')
                        if (p is not None):
                            p = p[0]
                        if (dep_row.iloc[m]['rel'] in ['nmod:poss', "obl:'s", "nmod:'s", "obl:’s", "nmod:’s"]):
                            p = 'of'
                        elif (p is not None):
                            p = dep_row.iloc[p]['lemma']
                        prepositions.append(p)
                    else:   #if no modifier, no preposition
                        prepositions.append(None)
            #2.6. translate indices into lemmatised tokens
            if (ent_mask is not None):
                gen_subjects = [dep_row.iloc[i]['lemma_gen'] for i in subjects]
                gen_objects = [(None if i==None else dep_row.iloc[i]['lemma_gen']) for i in objects]
                gen_modifiers = [(None if i==None else dep_row.iloc[i]['lemma_gen']) for i in modifiers]
            subjects = [dep_row.iloc[i]['lemma'] for i in subjects]
            verbs = [dep_row.iloc[i]['lemma'] for i in verbs]
            objects = [(None if i==None else dep_row.iloc[i]['lemma']) for i in objects]
            modifiers = [(None if i==None else dep_row.iloc[i]['lemma']) for i in modifiers]
            #prepositions already translated, otherwise => prepositions = [(None if i==None else dep_row.iloc[i]['lemma']) for i in prepositions]
            #2.7. add events of sentence to full list of events
            events = list(zip(*[subjects, verbs, objects, prepositions, modifiers])) #transpose
            all_events.append(events)
            if (ent_mask is not None):
                gen_events = list(zip(*[gen_subjects, verbs, gen_objects, prepositions, gen_modifiers]))
                all_gen_events.append(gen_events)
            events_count.append(len(subjects))
            pbar.update(1)
    plots['events'] = all_events
    if (ent_mask is not None):
        plots['events_generalised'] = all_gen_events
    plots['events_per_line'] = events_count
    pd.options.mode.chained_assignment = 'warn'

    #3. cleanup plots dataframe:
    plots.drop('dependencies', axis=1, inplace=True)
    #3.1. remove any line with no events
    plots.drop(plots[ plots['events_per_line']==0 ].index, inplace=True)
    #3.2. remove any entire summaries with total events (sum of len(events) per summary) < min_events 
    events_per_summary = {wid: plots.loc[wid]['events_per_line'].sum() for wid in plots.index.get_level_values(0).unique().values}
    wids_to_keep = [wid for wid in events_per_summary if events_per_summary[wid] >= min_events]
    plots = plots.loc[pd.IndexSlice[wids_to_keep,:]]
    plots.drop('events_per_line', axis=1, inplace=True)
    return plots

def simplify_reindex(plots, metadata):
    print("Cleaning index...")
    #1. simplify/translate wiki_id to story number in plots
    wids = plots.index.get_level_values(0).unique().values
    story_nums = range(len(wids))
    wid_to_story_num = dict(zip(wids, story_nums))
    plots.reset_index(inplace=True) #flatten
    plots['story'] = plots['wiki_id'].map(wid_to_story_num)
    plots.drop('wiki_id', axis=1, inplace=True)
    plots.set_index('story', inplace=True)
    #2. reset line numbers in plots (as lines have been removed)
    plots = plots.rename(columns={'line':'old_line'})
    plots.index = pd.MultiIndex.from_arrays( [
        plots.index.get_level_values(0), plots.groupby(level=0).cumcount()], names=['story', 'line'])
    plots.drop(columns='old_line', inplace=True)
    #3. remove discarded entries in metadata (stories removed due to lack of events)
    all_mwids = metadata.index.unique()
    drop_wids = list(set(all_mwids) - set(wids)) #wids: remaining after processing
    metadata.drop(drop_wids, inplace=True)
    #4. translate wiki_id to story number in metadata
    metadata.reset_index(inplace=True) #flatten
    metadata['story'] = metadata['wiki_id'].map(wid_to_story_num)
    metadata.drop('wiki_id', axis=1, inplace=True)
    #5. add empty entry where no metadata for a story (note: unnecessary for main dataset)
    #m_story_nums = metadata.index.values
    #m_add = pd.DataFrame([(sn,[]) for sn in set(story_nums) - set(m_story_nums)],
    #                     columns=['story','genres'])
    #metadata = pd.concat([metadata, m_add])
    metadata.set_index('story', inplace=True)
    metadata.sort_index(inplace=True)

    return plots, metadata

def add_synsets(plots, vague=True, drop_failures=False):
    print("Generalising unnamed entities and verbs...")
    pronouns = set(["I", "my", "me", "myself", "we", "us", "our", "ourselves", "self", "oneself", 
        "she", "her", "hers", "herself", "he", "him", "his", "himself",
        "they", "them", "their", "theirs", "themself", "themselves", "it", "its", "itself", 
        "you", "your", "yours", "yourself", "thou", "thee", "thy", "thine", "thyself",
        "who", "whom", "whomst", "whose", "whoever", "whosoever", "whomsoever",
        "which", "what", "this", "that", "these", "those", "whichever", "whatever",
        "someone", "somebody", "everyone", "everybody", "no-one", "noone", "nobody", "something", "anything"])
    pronouns_labels = {"male.n.02":set(["he", "him", "his", "himself"]), 
                       "female.n.02":set(["she", "her", "hers", "herself"]),
                       "person.n.01":set(["I", "my", "me", "myself", "you", "your", "yours", "yourself",
                                          "we", "us", "our", "ourselves", "theirself", "self", "oneself",
                                          "thou", "thee", "thy", "thine", "thyself",
                                          "who", "whom", "whomst", "whose", "whoever", "whosoever", "whomsoever",
                                          "someone", "somebody", "everyone", "everybody", "no-one", "noone", "nobody"]),
                       "physical_entity.n.01":set(["they", "them", "their", "theirs", "themselves"]),
                       "thing.n.08":set(["which", "what", "this", "that", "these", "those", 
                                         "whatever", "whichever", "something", "anything"])} #alt for physical_entity: "causal_agent.n.01"
    def gen_pronoun(p):
        for label in pronouns_labels:
            if p in pronouns_labels[label]:
                return label
        return 'entity.n.01'
    # nouns replaced by WordNet synset one levels up in hypernym hierarchy (but avoid making too general)
    def gen_noun(n):
        if (n is not None) and (n[:4] != "<NE>"):
            if n in pronouns:
                #n = "<PRN>"
                n = gen_pronoun(n)
            else:
                syns = wn.synsets(n, pos='n')
                if len(syns) == 0:
                    if drop_failures:
                        n = None
                    else:
                        n = 'entity.n.01'
                else:
                    syn = syns[0]
                    if vague and (len(syn.hypernyms()) != 0) and (syn.hypernyms()[0] not in syn.root_hypernyms()):
                        syn = syn.hypernyms()[0]
                    n = syn.name()
        return n
    # verbs ALSO replaced by wordnet class (verbnet bad)
    def gen_verb(v):
        if (v is not None):
            if v in pronouns:
                v = None        #delete event as erroneous
            else:
                syns = wn.synsets(v, pos='v')
                if len(syns) == 0:
                    if drop_failures:
                        v = None
                    else:
                        return v
                else:
                    syn = syns[0]
                    if vague and (len(syn.hypernyms()) != 0) and (syn.hypernyms()[0] not in syn.root_hypernyms()):
                        syn = syn.hypernyms()[0]
                    v = syn.name()
        return v
    def gen_obj(o): #obj can be either noun or complementary verb
        if (o is not None) and (o[:4] != "<NE>"):
            if (o[:4] == "<VB>"):
                o = gen_verb(o[4:])
            else:
                o = gen_noun(o)
        return o
    plots['events_generalised'] = plots['events_generalised'].progress_map(lambda events: [(gen_noun(e[0]), gen_verb(e[1]), gen_obj(e[2]), e[3], gen_noun(e[4])) for e in events])
    
    # drop all events with None subjects or None verbs
    plots['events_generalised'] = plots['events_generalised'].map(lambda events: [e for e in events if ((e[0] is not None) and (e[1] is not None))])
    wids = list(plots.index.get_level_values(0).unique())
    events_flat = [[e for event_line in plots.loc[wid, 'events'] for e in event_line] for wid in wids]
    events_gen_flat = [[e for event_line in plots.loc[wid, 'events_generalised'] for e in event_line] for wid in wids]
    n_events_before = [len(story) for story in events_flat]
    n_events_after = [len(story) for story in events_gen_flat]
    proportions_dropped = {wids[i]:(n_events_before[i] - n_events_after[i]) / n_events_before[i] for i in range(len(wids))}
    # if more than 20% of events in some singular plot are dropped, drop the whole plot
    drop_wids = [wid for wid in proportions_dropped if proportions_dropped[wid] > 0.2]
    plots.drop(drop_wids, level=0, inplace=True)
    print("Dropped average " + str(np.mean(list(proportions_dropped.values())) * 100) + "%% of events per story due to failed generalisation.")
    print("Dropped", len(drop_wids), "plots (" + str(100 * len(drop_wids) / len(wids)) + "%% of total).")
    return plots

def main():
    nltk.download('wordnet31')

    ####### OPTIONS #######
    romance_only = True
    generalise = True           # to do NER and generalise names, locations, nouns, verbs, etc.
    numbered_entities = True
    more_vague = False          # generalise nouns and verbs further
    drop_wn_failures = True     # discount words which failed to generalise with wordnet
    stop_at_n_plots = -1        # debug var: set to -1 to get all plots
    min_events_per_story = 10 
    dep_type = "enhancedPlusPlusDependencies"
    #######################

    if ((dep_type != "basicDependencies") and (dep_type != "enhancedDependencies") and (dep_type != "enhancedPlusPlusDependencies")):
        print("Parsing option dep_type must only take these values: 'basicDependencies', 'enhancedDependencies', or 'enhancedPlusPlusDependencies'.")
        exit()

    plots_raw, metadata = import_data(romance_only, stop_at_n_plots)
    plots = dependency_parse(plots_raw, stop_at_n_plots, dep_type)

    ent_mask = None
    if generalise:
        ent_mask, entities_per_plot, failed_wids = create_entity_mask(plots, stop_at_n_plots, numbered_entities)
        if len(failed_wids) > 0:
            print("NER failed for wiki_id:", failed_wids)
            plots.drop(failed_wids, level=0, inplace=True)
            metadata.drop(failed_wids, inplace=True, errors='ignore')
        metadata = pd.concat((metadata, entities_per_plot), axis=1)
    plots.drop('summary', axis=1, inplace=True)

    plots = eventify(plots, min_events=min_events_per_story, ent_mask=ent_mask)
    if generalise:
        plots = add_synsets(plots, vague=more_vague, drop_failures=drop_wn_failures)
    plots, metadata = simplify_reindex(plots, metadata)

    #export processed data
    nlp.close() 
    metadata.to_csv(os.path.join(data_folder, export_to, 'metadata.csv'), index=True)
    plots.to_csv(os.path.join(data_folder, export_to, 'plots.csv'), index=True)

    #export parameters
    params = {}
    params['generalised'] = generalise
    params['numbered_entities'] = numbered_entities
    params['more_vague'] = more_vague
    params['dep_type'] = dep_type
    with open(os.path.join(data_folder, export_to, 'dataset_params.txt'), 'w') as f:
        f.write(json.dumps(params))

    print("... done.")

if __name__ == '__main__':
    main()