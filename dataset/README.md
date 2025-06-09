# Dataset Information

## About the Dataset

This is a Language Detection dataset containing text samples from 17 different languages. The dataset consists of a total of 10,337 samples distributed across various languages with different representation levels.

| Language | Samples | Percentage | Example Text |
|----------|---------|------------|--------------|
| English | 1,385 | 13.40% | "do you mind if i jump in here?" |
| French | 1,014 | 9.81% | "pour moi en dessous de me dire lequel de ces 10 mots quel est votre préféré et aussi." |
| Spanish | 819 | 7.92% | "No, no quiero casarme contigo, quiero ir a casa." |
| Portuguese | 739 | 7.15% | "ou estúpido agora que você já ouviu a palavra obtuso em geometria..." |
| Italian | 698 | 6.75% | "in realtà sono le 13:00. proprio adesso." |
| Russian | 692 | 6.69% | "У меня почти высшая оценка - слово «громадный» имеет классические латинские корни..." |
| Swedish | 676 | 6.54% | "kan inte klaga på nästa fras någon säger till dig något..." |
| Malayalam | 594 | 5.75% | "ഉദാഹരണത്തിന് അവന്റെ മേശ മര്യാദ ക്രൂരമായിരുന്നു..." |
| Dutch | 546 | 5.28% | "ik red het wel." |
| Arabic | 536 | 5.19% | "إذا كان هذا هو ما تريده حقًا ، فسوف آخذك إلى المنزل ، حيث أخذ الفزع يبكي ويبكي." |
| Turkish | 474 | 4.59% | "nereden biliyorsunuz?." |
| German | 470 | 4.55% | "Kannst du bitte einen Moment warten?" |
| Tamil | 469 | 4.54% | "இறைச்சி மணமான மெலிசா அவள் உனக்கு என்ன வேண்டுமானாலும்..." |
| Danish | 428 | 4.14% | "siger "jeg ved det ikke"." |
| Kannada | 369 | 3.57% | "ನಾನು ನಿಮ್ಮೊಂದಿಗೆ 100% ಒಪ್ಪುತ್ತೇನೆ." |
| Greek | 365 | 3.53% | "αν κάτι σας εκπλήσσει, μπορείτε επίσης να πείτε ότι πρέπει να με αστειεύεσαι σαν θεέ μου." |
| Hindi | 63 | 0.61% | "[64] इस साइटने मीडिया के कई रूपों पर एक प्रभाव पैदा किया है..." |


## Class Imbalance Handling

- **Italian**: 698 samples (6.75%)
- **Non-Italian**: 9,639 samples (93.25%)

The original dataset exhibits severe class imbalance with a ratio of approximately 1:14 (Italian:Non-Italian). To address this imbalance and improve model performance, **under-sampling** is applied to the majority class (Non-Italian samples).

### Final Dataset:
- **Italian**: 698 samples (33.3%)
- **Non-Italian**: 1,396 samples (66.7%)
- **Total**: 2,094 samples
- **Final Ratio**: 1:2 (Italian:Non-Italian)