# Mechanism-Specific Cognitive Restoration in a Pruning-Based Computational Model of OCD: Implications for Ketamine, SSRIs, and Neurosteroids

Authors:

Ngo Cheung, FHKAM(Psychiatry)

Affiliations:

¹ Independent Researcher

Corresponding Author:

Ngo Cheung, MBBS, FHKAM(Psychiatry)

Hong Kong SAR, China

Tel: 98768323

Email: info@cheungngomedical.com

**Conflict of Interest**: None declared.

**Funding Declaration**: This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

**Ethics Declaration**: Not applicable.

## Abstract

Background: Obsessive-compulsive disorder (OCD) is associated with perseverative symptoms and persistent cognitive impairments, particularly in executive flexibility, set-shifting, and sensory gating, which often remain despite standard treatments. Excessive developmental synaptic pruning has been implicated as a core neurobiological vulnerability, yet comparative mechanistic insights into how different antidepressant classes address these cognitive deficits are limited.

Objective: To investigate mechanism-specific effects on cognitive restoration in OCD using a pruning-based recurrent neural network model of cortico-striato-thalamo-cortical loops, comparing ketamine-like rapid synaptogenesis, SSRI-like gradual functional tuning, and neurosteroid-like tonic inhibition.

Methods: A GRU-based recurrent network was trained on a rule-switching task, then subjected to 95% magnitude pruning to simulate OCD onset. From this shared vulnerable baseline, three interventions were applied and compared under an iso-dose framework (L1 weight-change norm). A multi-domain cognitive battery assessed flexibility (45° rotation probe), visuospatial processing (scaling), perceptual acuity (fine discrimination), distraction resistance (high noise), and sensory integration (conflicting blends at 25--75%). Performance was tracked across illness stages: healthy baseline, post-pruning onset, treatment phases (including mid-treatment for SSRI-like and off-medication for neurosteroid-like), and relapse (40% secondary pruning).

Results: Pruning reliably induced perseveration and broad cognitive deficits mirroring clinical OCD profiles. Ketamine-like structural regrowth achieved superior acute symptom relief and near-complete cognitive normalization, with greatest resilience to relapse. SSRI-like tuning was highly efficient but left substantial flexibility and integration deficits. Neurosteroid-like inhibition provided rapid intermediate benefits that partially reversed off-medication.

Conclusions: Structural synaptogenesis uniquely reverses pruning-induced cognitive inefficiencies in simulated OCD, suggesting ketamine-like agents may preferentially target executive and sensory burdens. Functional approaches offer efficient symptomatic control but limited restoration. These findings support personalized treatment sequencing guided by cognitive profile, with implications for rapid-acting interventions in cognition-impaired OCD subgroups.

## Introduction

Obsessive-compulsive disorder (OCD) affects roughly two to three percent of people worldwide and brings a heavy, long-term toll through intrusive thoughts and ritualistic behaviour \[1\]. First-line care---exposure-based cognitive-behavioural therapy combined with serotonin-reuptake inhibitors---relieves symptoms for many, yet close to half of treated patients continue to live with impairing residual signs \[2,3\]. Beyond the visible rituals, laboratory studies repeatedly document modest but widespread weaknesses in executive skills such as mental flexibility, response inhibition, and set-shifting, together with visuospatial and sensory-gating difficulties \[4,5\]. These cognitive lags often predict day-to-day functioning more accurately than clinical severity scores, and they seldom normalise after standard medication alone \[6\].

Pathophysiological models place the disorder within hyperactive cortico-striato-thalamo-cortical loops, where disrupted balances among excitatory, inhibitory, and neuromodulatory signals drive repetitive processing \[7\]. Developmental work further suggests that excessive synaptic pruning in adolescence may leave these circuits brittle, fostering the perseverative style seen in early-onset OCD and mirroring the selective cognitive profile noted in clinics \[8\].

Interest has therefore shifted toward agents that act outside classic serotonin mechanisms. Ketamine, an NMDA-receptor antagonist that boosts synaptic plasticity, delivers rapid relief in refractory depression and shows initial promise for anti-obsessional action \[9\]. Neurosteroids that enhance GABA_A signalling can also produce swift symptom reductions and may confer neuroprotective effects \[10,11\]. Rigorous head-to-head comparisons of these novel strategies with conventional serotonergic approaches are still rare, especially with regard to cognitive outcomes.

Computational modelling offers a controlled platform for probing such mechanistic questions. Earlier network simulations of affective illness reproduced pruning-related vulnerability and revealed that different drug classes drive distinct long-term paths \[12\]. Building on that groundwork, we constructed a gated recurrent network that approximates cortico-striatal circuitry, imposed severe developmental pruning (95 % sparsity), and then applied three contrasting interventions: (1) ketamine-like rapid, gradient-guided synaptogenesis, (2) SSRI-like gradual functional stabilisation through noise damping, and (3) neurosteroid-like broad inhibitory modulation. We normalised dose intensity across arms using an L1 weight-change metric and tracked performance on a multi-domain cognitive battery---covering flexibility, visuospatial skill, attention, and sensory integration---during active treatment, after washout, and throughout relapse cycles.

The present work tests whether pruning is a sufficient primary vulnerability for OCD-like phenomena, clarifies how specific pharmacological mechanisms remodel both symptoms and cognition, and aims to guide treatment sequencing when lingering executive problems restrict full recovery.

## Methods

### Computational framework and network architecture

![](media/image2.png){width="6.267716535433071in" height="3.388888888888889in"}

***Figure 1.** Computational Architecture of the CSTC Loop Model. The network processes 2D coordinate inputs through a simulated cortico-striato-thalamo-cortical (CSTC) loop. Blue nodes (Structural) represent synaptic weight matrices subject to pruning (OCD onset) and regrowth (Ketamine mechanism). Orange nodes (Functional) represent dynamic state modulations, including tonic inhibition (Neurosteroid mechanism) and stress-induced noise injection (SSRI mechanism). Purple node indicates the recurrent Gated Recurrent Unit (GRU) core responsible for maintaining working memory and rule representations. The architecture allows for the independent manipulation of structural connectivity and functional dynamics to simulate distinct therapeutic mechanisms.*

All experiments were carried out in PyTorch 2.9 on a CUDA-equipped workstation. The base model, intended to mimic cortico-striato-thalamo-cortical signalling (Figure 1), contained an input projection (two features to 128 units), a two-layer gated-recurrent unit with 64 hidden units per layer (0.1 dropout on the upper layer), and a linear read-out that mapped the 64-dimensional state to four class logits. ReLU activations followed the input layer, while neurosteroid simulations optionally added a tanh bound to recurrent states. Two modulators were built into every time-step update: additive Gaussian noise to represent stress and a multiplicative scaling term to impose pharmacological inhibition. Parameters were trained with Adam, gradient norms were clipped at 1, and a dedicated pruning manager maintained binary masks that enforced sparsity. For every data set and sweep the global random seed for NumPy and PyTorch was fixed at 42.

### Task and data generation

The network learned a sequential rule-switching problem that captures core demands of the Wisconsin Card Sorting Test. Two-dimensional inputs were drawn from four isotropic Gaussian clusters centred at (±1.5, ±1.5) with σ = 0.8. Labels were assigned by one of four rules---X sign, Y sign, quadrant or diagonal. Training sequences were 200 steps long with stochastic rule changes; test sequences retained the same length but switched rules deterministically every 50 trials. Each training epoch presented 500 sequences in mini-batches of 32; evaluation used 100 sequences.

### Cognitive probe battery

To track domain-specific effects, six derived test sets (50 sequences each) were administered at every illness stage. These probes rotated the stimulus space, magnified it, compressed cluster locations, added heavy sensory noise, injected strong internal stress or mixed in a conflicting \"voice\" stream at graded intensities. All probes preserved deterministic switch points so that flexibility measures were comparable.

### Simulation of OCD onset

After 50 epochs of baseline training at a learning rate of 0.001, synaptic resources were over-pruned by iterative magnitude pruning until only five percent of the original weights remained (Figure 2). This highly sparse model, which still performed the task, served as the common vulnerability state labelled \"OCD onset.\"

### Treatment protocols

From the shared onset network three intervention styles were launched. A ketamine-like procedure first regrew between 10 % and 80 % of previously pruned weights using gradient-based importance scores collected over five batches; a brief consolidation phase of 10--30 epochs (learning rate 5 × 10⁻⁴) followed. An SSRI-like arm left the architecture unchanged but trained for 20--200 additional epochs while linearly damping internal noise from 0.4 to 0.0; learning rates ranged from 1 × 10⁻⁶ to 5 × 10⁻⁵. A neurosteroid-like condition imposed constant inhibitory scaling of the hidden state (0.50--0.85) with optional tanh bounding and then ran eight consolidation epochs. When neurosteroid effects were evaluated off-medication the scaling factor was simply removed.

### Iso-dose matching

To compare mechanisms fairly, \"dose\" was defined as the mean absolute weight change (L1 norm) relative to the pruned baseline. Parameter sweeps generated full dose--response curves, and configurations that fell within ±0.002 of target doses (0.005 and 0.010) were selected for head-to-head analysis. Supplementary indices---the L2 norm of changes, proportion of weights with more than ten percent deviation, and the resulting sparsity---were logged for completeness.

### Evaluation metrics and timing

Primary outcomes were overall classification accuracy, perseverative error rate, switch cost, a flexibility index (accuracy during the first ten steps after a rule change divided by accuracy during stable periods) and trials-to-recovery after each switch. The cognitive probe battery was run at ten checkpoints: healthy baseline, immediately after pruning, pre-treatment, mid-treatment for the SSRI arm (25 %, 50 %, 75 % of epochs), immediate drug effect for neurosteroid, post-treatment, neurosteroid off-medication, pre-relapse and post-relapse. Relapse risk was induced by pruning 40 % of surviving weights once treatment ended, after which the full test suite was repeated.

![](media/image3.png){width="4.640199037620297in" height="7.629899387576553in"}

*Figure 2. Experimental Pipeline for Multi-Mechanism Comparison. The simulation proceeds through four distinct phases. Phase 1 establishes the disease model by training a healthy network and applying magnitude-based pruning to simulate OCD onset. Phase 2 branches into three distinct therapeutic mechanisms: structural regrowth (Ketamine), functional stress reduction (SSRI), and tonic inhibition (Neurosteroid). Phase 3 ensures fair comparison via \"Iso-Dose\" matching, where treatment parameters are aligned based on the L1 norm of total synaptic weight change. Phase 4 assesses long-term stability through a relapse simulation involving secondary pruning. Comprehensive cognitive batteries (Probes 1-4) are administered at every stage to track deficits in flexibility, attention, and sensory integration.*

All numerical results reported in this paper come from the seed-42 run; however every sweep covered the full parameter range in order to locate optimal and iso-dose operating points.

## Results

### Pruning-Induced OCD-Like Phenotype

Reducing synaptic weights to 5 % of their original number created a network that behaved like an untreated OCD system. Overall accuracy on the standard task collapsed from 0.76 in the healthy model to 0.31 after pruning. Perseverative errors showed the opposite shift, jumping from 0.28 to 0.85. Comparable losses were recorded on every cognitive probe: rotation accuracy fell to 0.26, scaling to 0.32, fine discrimination to 0.30, high-noise performance to 0.31, and the 50 % sensory-blend condition to 0.29. Together these figures confirm that extreme pruning alone is sufficient to reproduce the hallmark rigidity, distractibility, and interference sensitivity seen in clinical OCD.

### Acute Treatment Outcomes

Starting from the shared pruned baseline, each pharmacological analogue produced a distinct recovery pattern. A ketamine-like pulse that regrew 60 % of the deleted connections (L1 dose = 0.0116) raised accuracy to 0.75, cut perseveration to 0.24, and lowered effective sparsity to 38 %. By contrast, the SSRI-like protocol, which altered weights only minimally (dose = 0.0009), lifted accuracy to 0.46 and left perseveration high at 0.75. Neurosteroid-style tonic inhibition (dose = 0.0020) sat between the two: accuracy reached 0.56 and perseveration 0.65 while the drug was on board, then drifted to 0.67 when inhibition was removed. Expressed as perseveration improvement per unit dose, the SSRI routine appeared most \"efficient\" (139.6), followed by neurosteroids (114.9) and ketamine (55.2).

### Post-Treatment Cognitive Profiles

Domain testing revealed wider gaps (Table 1). After ketamine-like repair, rotation accuracy returned to 0.48--0.49, scaling and high-noise probes landed near their healthy baselines (about 0.76 and 0.75), and the 50 % sensory-blend condition climbed to 0.66---virtually indistinguishable from the intact network. SSRI-treated models recovered little flexibility (rotation 0.27) and only partial gains on other probes (scaling 0.45, blend 0.42). Neurosteroid inhibition produced solid but reversible benefits: rotation 0.31, scaling 0.56, fine discrimination 0.52, high noise 0.55, and blend 0.51 while active, with mild deterioration once the inhibitor was withdrawn.

***Table 1.** Post-Treatment Cognitive Probe Accuracy*

| **Condition**          | **Rotation** | **Scaling** | **Fine Disc** | **High Noise** | **Blend 50%** |
|------------------------|--------------|-------------|---------------|----------------|---------------|
| Healthy Baseline       | 0.5339       | 0.7655      | 0.7269        | 0.7602         | 0.6542        |
| OCD Onset              | 0.2633       | 0.3153      | 0.3049        | 0.3101         | 0.2932        |
| Ketamine-like          | 0.4837       | 0.7601      | 0.7101        | 0.7494         | 0.6563        |
| SSRI-like              | 0.2657       | 0.4518      | 0.4600        | 0.4450         | 0.4166        |
| Neurosteroid-like (on) | 0.3098       | 0.5552      | 0.5198        | 0.5458         | 0.5134        |

### Relapse Vulnerability and Cognitive Impact

A second pruning event that removed 40 % of surviving synapses served as a relapse trigger (Table 2). Networks previously exposed to ketamine showed the greatest resilience: global accuracy slipped by 0.03, perseveration rose just 0.02, rotation accuracy declined 0.16, and blend accuracy changed less than 0.01. SSRI models lost 0.06 in accuracy, gained 0.06 in perseveration, and dropped 0.08 on fine discrimination; interestingly, their rotation score improved by 0.07, signalling unstable adaptation rather than true robustness. Neurosteroid models were most fragile: accuracy fell 0.12, perseveration increased 0.16, and large losses appeared in fine discrimination (-0.11) and blend (-0.08), even though rotation rose slightly (+0.09).

***Table 2.** Cognitive Changes Post-Relapse*

| **Treatment**     | **ΔAccuracy** | **ΔRotation** | **ΔFine Disc** | **ΔBlend 50%** |
|-------------------|---------------|---------------|----------------|----------------|
| Ketamine-like     | -0.0298       | -0.1577       | -0.0440        | -0.0069        |
| SSRI-like         | -0.0620       | +0.0710       | -0.0827        | -0.0320        |
| Neurosteroid-like | -0.1156       | +0.0899       | -0.1082        | -0.0798        |

### Iso-Dose Matched Comparisons

Parameter sweeps allowed direct comparison at fixed network change (Table 3). When all arms were tuned to an L1 shift of roughly 0.005, ketamine preserved the lead: perseveration 0.34, rotation 0.41, blend 0.65. Neurosteroids followed (perseveration 0.67, rotation 0.30, blend 0.51) and SSRIs lagged (perseveration 0.70, rotation 0.29, blend 0.47). The same rank order held at a 0.010 dose, with the ketamine gap widening.

***Table 3.** Iso-Dose Cognitive Outcomes (Target L1 ≈0.005)*

| **Treatment**     | **Actual Dose** | **Acute Perseveration** | **Rotation Accuracy** | **Blend 50% Accuracy** |
|-------------------|-----------------|-------------------------|-----------------------|------------------------|
| Ketamine-like     | 0.0045          | 0.338                   | 0.406                 | 0.645                  |
| SSRI-like         | 0.0014          | 0.695                   | 0.288                 | 0.470                  |
| Neurosteroid-like | 0.0022          | 0.667                   | 0.302                 | 0.507                  |

Taken together, these findings show that:

- Severe pruning alone can generate an OCD-like behavioural signature.

- Ketamine-style structural repair delivers the broadest and most durable cognitive recovery, though at the highest energetic cost.

- SSRI-style tuning is metabolically parsimonious but leaves large executive gaps and confers limited protection against further damage.

- Neurosteroid-style inhibition offers fast, intermediate relief that is contingent on continued drug presence.

## Discussion

### Interpretation of model findings in relation to OCD neurobiology and treatment response

The extreme pruning step reproduced the clinical picture of obsessive-compulsive disorder with surprising fidelity. Removing 95 % of synapses created a network that repeated old responses, struggled to shift rules, and failed when distracting information was introduced. Those patterns mirror meta-analytic evidence of moderate but selective executive weaknesses in patients---especially in flexibility and sensory gating---while basic attention remains intact \[4,5\]. The model therefore supports the idea that OCD reflects inefficient, not globally damaged, circuitry \[6\].

When different pharmacological analogues were applied, their effects echoed emerging clinical data. The ketamine-like protocol, which rebuilt connections quickly, almost normalised behaviour across probes and protected the network from a second \"relapse\" pruning. This dovetails with proof-of-concept trials showing that a single ketamine infusion can relieve intrusive thoughts within minutes and for at least several days \[9\]. By fostering new synapses, the drug appears to restore the reserve lost to excessive developmental pruning.

The SSRI-like routine achieved respectable symptom relief at a fraction of the structural cost, yet left large gaps in rule-shifting and interference control. Clinically, serotonergic drugs reduce Yale--Brown scores but seldom erase executive deficits, even in full responders \[5\]. The simulation suggests this is because functional noise-damping stabilises an already thinned network rather than replacing lost resources.

Neurosteroid-style tonic inhibition occupied middle ground. While active, it enhanced accuracy and diminished perseveration; however, the benefits dissipated following the removal of the inhibitor, and relapse sensitivity persisted at elevated levels. The state-dependent profile fits with neurosteroid trials for mood disorders that work quickly but need to be taken regularly \[11\] and with research showing that endogenous GABAergic tone changes when you\'re under stress \[10\].

Iso-dose comparisons clarified trade-offs. Even when matched on mean absolute weight change, ketamine-like repair delivered the strongest cognitive recovery, especially on set-shifting and sensory-blend probes. SSRIs showed the highest efficiency---symptom change per unit weight shift---but the weakest executive gains, reinforcing clinical observations that they treat downstream manifestations, not the underlying synaptic scarcity. Neurosteroids sat between these poles, offering rapid relief that remained contingent on continued modulation.

Altogether, the model portrays OCD as a disorder of synaptic shortage rather than a simple serotonergic fault. Interventions that rebuild circuitry appear uniquely capable of correcting entrenched cognitive inefficiency, whereas modulatory agents provide economical symptom control but limited restoration of flexibility.

### Clinical guidelines and implications

Current practice still starts with high-dose selective serotonin reuptake inhibitors and exposure-and-response-prevention therapy \[13,14\]. These strategies help roughly half of patients, yet many continue to struggle with set-shifting and interference control even after obsessions and compulsions ease \[4\]. The present simulations give one possible reason: an SSRI-like \"functional tuning\" settles the traffic inside an already thinned circuit but cannot rebuild the synaptic reserve that flexible thinking requires (Table 4). In other words, there is a ceiling on how far serotonin alone can take recovery.

By contrast, the ketamine-like intervention triggered rapid synapse growth and produced broad cognitive gains that survived a relapse challenge. These modelling results fit well with small clinical trials showing that a single ketamine infusion can cut obsessive thoughts within minutes and keep them down for days \[9\]. They also hint that plasticity-enhancing drugs might deserve a place earlier in care pathways, especially for patients whose main handicap is cognitive rigidity. Using ketamine as an induction or augmentation while cognitive-behavioural therapy gets started could be explored in future work.

Neurosteroid-like modulation landed between the two poles. While the agent was present, symptoms and cognition improved at low network \"cost,\" but benefits slipped once the drug was withdrawn. This pattern echoes the phasic swings of endogenous neurosteroids and the maintenance-dependent effects reported with synthetic analogues such as zuranolone \[11\]. Hence, short courses might be handy for flare-ups or for people who cannot tolerate structural interventions, although longer regimens would likely be needed to keep gains.

Interestingly, these recommendations reverse the hierarchy seen in our earlier bipolar-depression model, where inhibitory neurosteroid-like actions worked best for maintenance and ketamine-like repair was reserved for crisis states \[15\]. In the OCD framework, the core problem appears to be synaptic scarcity from the outset, so rebuilding circuitry offers both acute and lasting benefit, whereas purely modulatory approaches look more palliative. Such disorder-specific contrasts underline the value of computational assays when clinicians weigh treatment sequences across diagnoses.

Taken together, the findings support a mechanism-based sequence.

- SSRI-like agents remain sensible first-line choices for milder cases or clearly serotonin-responsive presentations because they are efficient and carry low risk.

- Ketamine-like plasticity enhancers may suit patients whose day-to-day disability stems from cognitive rigidity or sensory gating problems.

- Neurosteroid-like modulators can provide rapid help with minimal disruption, but sustained dosing is probably required.

Routine cognitive screening that looks beyond symptom counts---focused on tasks tapping set-shifting and interference resolution---could help match individuals to these options and move care closer to true personalisation.

***Table 4.** Comparative Clinical Implications of Pharmacological Mechanisms in OCD Modeling*

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 20%" />
<col style="width: 25%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr class="header">
<th><strong>Intervention Class</strong></th>
<th><strong>Modeled Mechanism</strong></th>
<th><strong>Key Simulation Findings</strong></th>
<th><strong>Clinical Recommendation</strong></th>
</tr>
<tr class="odd">
<th><p>SSRI-like</p>
<p><em>(Serotonergic)</em></p></th>
<th>Functional tuning; settling network traffic without structural change.</th>
<th>Efficient symptom reduction but limited by a "ceiling effect"; fails to restore synaptic reserve needed for cognitive flexibility.</th>
<th>Remains a practical first-line therapy for milder cases or clearly serotonin-responsive presentations due to high efficiency and low risk profile.</th>
</tr>
<tr class="header">
<th><p>Ketamine-like</p>
<p><em>(Glutamatergic)</em></p></th>
<th>Rapid synapse growth; structural rebuilding of lost connections.</th>
<th>Produced broad cognitive gains (restoring set-shifting/gating) that persisted through relapse challenges.</th>
<th>Best suited for patients with cognitive rigidity or sensory gating deficits. Potential use as induction/augmentation alongside CBT.</th>
</tr>
<tr class="odd">
<th><p>Neurosteroid-like</p>
<p><em>(GABAergic)</em></p></th>
<th>Phasic modulation; transient inhibitory damping.</th>
<th>Rapid improvement at low network cost, but benefits fade immediately upon withdrawal (maintenance-dependent).</th>
<th>Indicated for acute flare-ups or patients intolerant to structural interventions. Likely requires sustained dosing to maintain gains.</th>
</tr>
</thead>
<tbody>
</tbody>
</table>

*Note: Comparison contrasts with previous bipolar disorder models \[15\], where inhibitory modulation was superior for maintenance. In OCD, structural repair (Ketamine-like) appears necessary to overcome baseline synaptic scarcity.*

### Novelty and potential impact

Our modelling strategy breaks fresh ground in several ways (Figure 3). Earlier pruning studies focused on schizophrenia or mood disorders, but applying a cortico-striato-thalamo-cortical network to obsessive-compulsive disorder and then testing three very different drug classes side by side has not, to our knowledge, been attempted. Matching the interventions by the same amount of total weight change (the L1 dose) created a level playing field, showing clearly that rebuilding lost connections---a ketamine-like action---restores cognitive flexibility and sensory gating more fully than either gradual serotonergic retuning or broad inhibitory damping. Tracking a full probe battery from healthy training through onset, treatment, wash-out and relapse also offered a rare view of how specific cognitive domains wax and wane across the illness course.

![](media/image1.png){width="6.268099300087489in" height="6.0in"}

***Figure 3.** Novelty and Potential Clinical Impact. The study introduces a novel computational framework for OCD by applying a CSTC pruning model to compare three distinct pharmacological mechanisms. Methodological innovations include \"Iso-Dose\" matching to ensure fair comparison of synaptic weight changes and a longitudinal cognitive battery to track deficits across the illness course. The results provide specific clinical implications: glutamatergic agents (Ketamine-like) are identified as optimal for restoring structural flexibility in rigid phenotypes; SSRIs remain supported as efficient first-line therapies; and neurosteroids are highlighted for acute symptom management. Broader impacts include the demonstration of circuit specificity, contrasting these findings with previous bipolar disorder models.*

These advances matter clinically. The strong recovery produced by the ketamine analogue supports growing interest in glutamatergic agents for patients whose main difficulty is executive rigidity \[9\]. At the same time, the high efficiency but modest cognitive reach of the SSRI analogue reinforces the continuing role of optimised serotonin reuptake inhibitors as practical first-line therapy. The neurosteroid pattern---rapid relief that fades when the drug is removed---suggests a place for short, symptom-targeted use in acute flares. Finally, the sharp contrast with our earlier bipolar model, where inhibitory modulation proved most protective \[12\], underlines that circuit pathology differs across disorders; \"one-size\" mechanisms are unlikely to fit all.

### Limitations

Important caveats remain. The rule-switching task is only a pared-down stand-in for the emotional and social challenges faced by people with OCD; richer tasks could change the balance of results \[6\]. Running the simulations with a single random seed limits confidence in the stability of the findings, and no patient data were used to calibrate effect sizes. The pharmacological surrogates are necessarily stylised---ketamine as pure synaptic regrowth, neurosteroids as uniform inhibition---leaving out receptor kinetics, downstream signalling, and interactions with psychotherapy or comorbidity. Finally, relapse was mimicked by a second pruning event, a simplification that omits inflammatory or glial processes that may contribute to real-world neuroprogression.

### Conclusion

Within these boundaries, the model captures the core perseverative behaviour and selective cognitive burdens of obsessive-compulsive disorder and shows why structural repair may beat functional modulation when the fundamental problem is synaptic scarcity. These insights argue for tailoring treatment to mechanism: use serotonin reuptake inhibitors when symptoms are mild and circuits largely intact, turn to plasticity-enhancing strategies when rigidity dominates, and reserve rapid inhibitory agents for short-term relief. Treatments that both control symptoms and build lasting cognitive resilience are more likely to work if they rebuild circuit capacity instead of just slowing down activity.

## References

\[1\] Ruscio AM, Stein DJ, Chiu WT, et al. The epidemiology of obsessive-compulsive disorder in the National Comorbidity Survey Replication. Mol Psychiatry. 2010;15(1):53--63.

\[2\] Pallanti S, Quercioli L. Treatment-refractory obsessive-compulsive disorder: methodological issues, operational definitions and therapeutic lines. Prog Neuropsychopharmacol Biol Psychiatry. 2006;30(3):400--412.

\[3\] Soomro GM, Altman D, Rajagopal S, et al. Selective serotonin re-uptake inhibitors (SSRIs) versus placebo for obsessive compulsive disorder (OCD). Cochrane Database Syst Rev. 2008;(1):CD001765.

\[4\] Snyder HR, Kaiser RH, Warren SL, et al. Obsessive-compulsive disorder is associated with broad impairments in executive function: A meta-analysis. Clin Psychol Sci. 2015;3(2):301--330.

\[5\] Shin NY, Lee TY, Kim E, et al. Cognitive functioning in obsessive-compulsive disorder: A meta-analysis. Psychol Med. 2014;44(6):1121--1130.

\[6\] Abramovitch A, Abramowitz JS, Mittelman A. The neuropsychology of adult obsessive--compulsive disorder: A meta-analysis. Clin Psychol Rev. 2013;33(8):1163--1171.

\[7\] Milad MR, Rauch SL. Obsessive-compulsive disorder: Beyond segregated cortico-striatal pathways. Trends Cogn Sci. 2012;16(1):43--51.

\[8\] Rosenberg DR, MacMaster FP, Keshavan MS, et al. Decrease in caudate glutamatergic concentrations in pediatric obsessive-compulsive disorder patients taking paroxetine. J Am Acad Child Adolesc Psychiatry. 2000;39(9):1096--1103.

\[9\] Rodriguez CI, Kegeles LS, Levinson A, et al. Randomized controlled crossover trial of ketamine in obsessive-compulsive disorder: Proof-of-concept. Neuropsychopharmacology. 2013;38(12):2475--2483.

\[10\] Pinna G. In a mouse model relevant for post-traumatic stress disorder, selective brain steroidogenic stimulants (SBSS) improve behavioral deficits by normalizing allopregnanolone biosynthesis. Behav Pharmacol. 2010;21(5-6):438--450.

\[11\] Gunduz-Bruce H, Silber C, Kaul I, et al. Trial of SAGE-217 in Patients with Major Depressive Disorder. N Engl J Med. 2019;381(10):903--911.

\[12\] Cheung N. Irreversible episode-induced scarring and differential repair in simulated bipolar disorder progression \[Preprint\]. Zenodo. 2026a. https://doi.org/10.5281/zenodo.18304566

\[13\] Koran LM, Hanna GL, Hollander E, et al. Practice guideline for the treatment of patients with obsessive-compulsive disorder. Am J Psychiatry. 2007;164(7 Suppl):5--53.

\[14\] National Institute for Health and Care Excellence. 2019 surveillance of obsessive-compulsive disorder and body dysmorphic disorder: Treatment (NICE guideline CG31). NCBI Bookshelf. 2019 Feb 27. https://www.ncbi.nlm.nih.gov/books/NBK551808/

\[15\] Cheung N. Structural synaptogenesis superior to functional modulation in a pruning-based recurrent network model of OCD. Zenodo. 2026b. https://doi.org/10.5281/zenodo.18337490
