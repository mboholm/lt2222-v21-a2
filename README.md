# LT2222 V21 Assignment 2

Assignment 2 from the course LT2222 in the University of Gothenburg's winter 2021 semester.

Your name: Max Boholm (gusbohom)

## On preprocessing
Words of the dataset are:
1. transformed into lower-case; and
2. lemmatized using NLTK's `WordNetLemmatizer`. 

3. Punctuations are removed. 

## On instance representation and feature extraction
For every named entity (NE), features are extracted. By default, features are defined as *words* in the context C of NE, where C is restricted as follows:

1. five words left of NE, and
2. five words right of NE, but
3. only words in the same sentence as NE.

For example, consider:

> So they went outside. And Pooh looked at the knocker and the notice below it, and he looked at the bell-rope and the notice below it, and the more he looked at the bell-rope, the more he felt that he had seen something like it, somewhere else, sometime before.
> (https://winnie-the-pooh.bib.bz/chapter-4-in-which-eeyore-loses-a-tail-and-pooh-finds-one)

For NE *Pooh* the features are (lemmatized):

    [and, look, at, the, knocker, and]

For Bonus B, C is extended to include:

4. part-of-speech tags for the words in C.

## On backtracking instances
Bonus Part A asks for an analysis of instances being classified incorrectly. Pandas Data Frames and its method `index` are useful when backtracking false predictions. However, the Jupyter Notebook calls `test_y[0]` which returns a key error if `test_y` is a Pandas Data Frame. Therefore, the default setting is to return the output of `ttsplit` (`test_y` etc.) as Numpy arrays. In order to use the same functions to backtrack instances of NE class *org*, `backtrack` can be set to `True` and calling `finding_neverland` as: 

    a2.finding_neverland(test_y, test_predictions, "org", instances)

## Bonus Part A - Error Analysis 
> Look at the weakest-performing classes in the confusion matrix (or any, if they all perform poorly to the same extent). Find some examples in the test data on which the classifier classified incorrectly for those classes. What do you think is the reason why those are hard? Consider linguistic factors and statistical factors, if applicable.

The worst performing NE class is "org", standing for *organization* (it seems, from looking at the GMB dataset). Inspecting instances being identified as "org" in the original dataset (i.e. "true" *org*:s), which has been classified otherwise, suggests that these have other NEs in their context. Thus, the statistical reason for the poor performance in classifying *org* is its frequent close co-occurence with other classes. Accordinly, *org* shares many of its features with other classes (e.g. *geo* and *per*) and it will be confused with those classes. This situation makes some intuitive sense linguistically, although this thesis requires further study to be confirmed: organizations tend to be spoken/written about in relation to their location and the persons representing them (or otherwise related to them). Other named entities, such as persons and geographical places are used more independent of other named entities. 

An random set of examples where instances of *org* that have been classified as something else, where my preliminary named entity analysis is indicated by bold:

* Class: org Features: [**'czech', 'republic'**, **'russia'**, 'take', **'slovakia'**, 'and', **'the', 'united', 'state'**]
* Class: org Features: [**'reporter', 'without', 'border'**, 'and', 'association', 'welcome', 'the', 'release', 'of']
* Class: org Features: ['the', 'annual', 'report', 'of', 'council', 'know', 'as', **'abac'**]
* Class: org Features: ['newspaper', 'limited', 'which', 'own', 'time', 'say', 'it', 'be', 'provide']
* Class: org Features: ['exceptional', 'member', 'of', 'the', 'party', 'immediately', 'challenge', 'the', 'decision']
* Class: org Features: ['under', 'attack', 'from', **'taliban'**, **'al-qaida'**, 'militant']
* Class: org Features: [**'nasa'**, 'official', 'now', 'say', 'the']
* Class: org Features: ['a', 'muslim', 'organization', **'the', 'council', 'of', 'the', 'muslim', 'faith'**]
* Class: org Features: ['nominate', 'and', 'confirm', 'by', **'senate'**, **'hill'**, 'would', 'replace', 'another']
* Class: org Features: ['election', 'brotherhood', 'have', 'triple', 'its', 'strength']
* Class: org Features: ['region', **'u.n.'**, 'refugee', **'agency'**, 'say', 'international']
* Class: org Features: ['bomb', 'the', 'director', 'of', 'advocate', 'say', 'it', 'still', 'have']
* Class: org Features: ['nine', '**afghan**', 'soldier', 'and', 'taleban', 'rebel', 'have', 'be', 'kill']
* Class: org Features: ['**al-qaida**', "'s", 'deputy', 'leader', '**ayman**']
* Class: org Features: ['**iran**', 'be', 'refer', 'to', 'council']
* Class: org Features: ['suppose', 'to', 'have', 'be', 'nation', 'safe', 'haven']
* Class: org Features: ['agency', **'u.n.'**, 'say', 'the', 'deal', 'which']
* Class: org Features: ['can', 'move', 'to', 'the', 'house', 'as', 'early', 'as', **'september'**]
* Class: org Features: ['military', 'commander', 'in', **'afghanistan'**, 'taleban', 'insurgent', 'could', 'stage', 'a']
* Class: org Features: ['time', 'away', 'from', **'the', 'congress'**]
