# Machine learning classification of stress intervention following cognitive load tasks

This project uses machine learning algorithms to classify stress conditions from Empatica E4 data from Thayer & Stevens (2021).

### Overview

In [Thayer & Stevens (2021)](https://www.apa-hai.org/haib/download-info/effects-of-human-animal-interactions-on-affect-and-cognition/), we conducted two experiments investigating the effects of human-animal interaction (HAI) on affect and cognition. In these experiments, participants experienced cognitive (working memory, attentional control) tasks before and after either a three-minute exposure to a dog (HAI condition) or a control task. Self-report measures of affect (mood, anxiety, stress) were collected repeatedly throughout the sessions. While interacting with a dog influenced measures of affect, they did not influence cognition. In addition to the affect and cognition measures, participants wore an [Empatica E4](https://www.empatica.com/research/e4/) to record heart rate, electrodermal activity, body temperature, etc. However, we did not analyze these data, as the sampling rate was too low for robust measures of heart rate variability ([Malik et al. 1996](https://doi.org/10.1093/oxfordjournals.eurheartj.a014868), [Laborde et al. 2017](https://doi.org/10.3389/fpsyg.2017.00213)).

[Arce & Gehringer (2021)](https://doi.org/10.21105/joss.03455) have investigated machine learning algorithms that take Empatica E4 data and classify whether participants were experiencing stress induction ([GitHub repo](https://github.com/Munroe-Meyer-Institute-VR-Laboratory/Biosensor-Framework)). This offers the opportunity to apply the machine learning algorithms developed to classify stress to the Thayer & Stevens (2021) data and address two specific aims:

### Specific aims

1. Assess the predictive accuracy of the machine learning algorithms in classifying intervention exposure
2. Generate an index of physiological stress from Empatica E4 output

### Research questions

1. How accurately can machine learning algorithms classify stress intervention exposure?
1. Does accuracy differ between stress reduction interventions and stress induction activities?
1. Which algorithms best predict stress intervention exposure?
1. Which features best predict passive and active stress intervention exposures?
1. Do continuous outcome measures from the algorithms differ across exposure conditions?
