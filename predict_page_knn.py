from sklearn.neighbors import NearestNeighbors
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MaxAbsScaler


def load_model():
    anime = pd.read_csv(r"anime.csv")
    return anime



anime = load_model() 
    

def show_predict_page_knn():
    st.title("Anime Recommendation System")
    st.write("We need some information to pridict ")

    df = pd.read_csv(r"a1.csv",usecols=["name"])
    
    anime_name = st.selectbox("Anime_name :: ", (df))
    ok = st.button("Make prediction")
    if ok:
        anime.loc[(anime["genre"] == "Hentai") & (anime["episodes"] == "Unknown"), "episodes"] = "1"
        anime.loc[(anime["type"] == "OVA") & (anime["episodes"] == "Unknown"), "episodes"] = "1"
        anime.loc[(anime["type"] == "Movie") & (anime["episodes"] == "Unknown")] = "1"

        known_animes = {"Naruto Shippuuden": 500, "One Piece": 784, "Detective Conan": 854, "Dragon Ball Super": 86,
                        "Crayon Shin chan": 942, "Yu Gi Oh Arc V": 148, "Shingeki no Kyojin Season 2": 25,
                        "Boku no Hero Academia 2nd Season": 25, "Little Witch Academia TV": 25}

        for k, v in known_animes.items():
            anime.loc[anime["name"] == k, "episodes"] = v

        anime["episodes"] = anime["episodes"].map(lambda x: np.nan if x == "Unknown" else x)
        
        anime["episodes"].fillna(anime["episodes"].median(),inplace = True)
        anime["rating"] = anime["rating"].astype(float)
        anime["rating"].fillna(anime["rating"].median(),inplace = True)
        
        pd.get_dummies(anime[["type"]])
        anime["members"] = anime["members"].astype(float)

        anime_features = pd.concat([anime["genre"].str.get_dummies(sep=","), pd.get_dummies(
            anime[["type"]]), anime[["rating"]], anime[["members"]], anime["episodes"]], axis=1)

        anime["name"] = anime["name"].map(
            lambda name: re.sub('[^A-Za-z0-9]+', " ", name))
        
        anime_features.columns

        max_abs_scaler = MaxAbsScaler()
        anime_features = max_abs_scaler.fit_transform(anime_features)

        nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(anime_features)
        distances, indices = nbrs.kneighbors(anime_features)

        all_anime_names = list(anime.name.values)

        def get_index_from_name(name):
            return anime[anime["name"]==name].index.tolist()[0]
        
        all_anime_names = list(anime.name.values)

        def get_id_from_partial_name(partial):
            for name in all_anime_names:
                if partial in name:
                    print(name,all_anime_names.index(name))

        def print_similar_animes(query=None,id=None):
            l = []
            if id:
                for id in indices[id][1:]:
                    l.append(anime.iloc[id]["name"])

            if query:
                found_id = get_index_from_name(query)
                for id in indices[found_id][1:]:
                    l.append(anime.iloc[id]["name"])

            return l
        st.subheader("Recommended animes are :: ")
        rec = print_similar_animes(anime_name)
        for i in rec:
            st.write(i)
    else:
        pass


