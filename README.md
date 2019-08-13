# Overview

A hackathon project initiated at [MIND 2019](https://mindsummerschool.org/current_year.html) at Dartmouth College.

High-level idea: create narrative trajectories (kind of like [these](https://github.com/ContextLab/sherlock-topic-model-paper)) for movie scripts mined from IMDB.  Then relate various aspects of those trajectories and/or the contents of the scripts to interesting things like:
- movie ratings and reviews
- genre
- box office success (money made, number of tickets sold, etc.)
- etc.

# Setup

To initialize the `cluster-tools` module and switch to the `eventseg` branch (needed to run the analyses on the Discovery cluster), run the following (in Terminal, from within the narrative_complexity directory):

```
git submodule update --init
cd code/cluster-tools-dartmouth
git checkout eventseg
```

Install all project requirements by running (from within the project folder):
```
pip install -r requirements.txt
```

Then run (from within `Jupyter Lab` or a `Jupypter notebook`) any of the `.ipynb` files.

# Presentation

Our hackathon presentation may be found [here](https://docs.google.com/presentation/d/1pWY2tV-5vg0wTS-ujJM0wq9TBW_hfLN1kz_Bud37cR0/edit?usp=sharing).

# Brainstorm

Potential directions for the project
- Create a database of semantic content of movies that
    people might want to refer to in order to select
    an ideal stimulus space for functional alignment
- Analyze ISC across naturalistic datasets to assess
    whether narrative complexity is related to similarities
    in functional topography
- Use narrative complexity to predict ratings and gross 
    value of films
- Assess consistency of the narrative across different 
    segments of movies
- Compare consistency of movie content and critical reviews
