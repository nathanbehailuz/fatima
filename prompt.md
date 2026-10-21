For the **Amharic condition**, keep the instructions in Amharic and explicitly allow A–D as answer labels:

```text
ከታች ያለውን ጥያቄ እና አራቱን አማራጮች በጥንቃቄ አንብብ።

1. ከአማራጮቹ አንዱን ብቻ ምረጥ።
2. በመጀመሪያው መስመር የመረጥከውን አማራጭ ፊደል
   (A፣ B፣ C ወይም D) ብቻ ጻፍ።
3. ከዚያ በኋላ፣ የመረጥከው መልስ ለምን ትክክል እንደሆነ
   በአማርኛ ከ2 እስከ 4 ዓረፍተ ነገሮች አብራራ።
   ማብራሪያው መልሱን ብቻ የሚደግም ሳይሆን፣
   ለምርጫህ ምክንያት የሚሰጥ ይሁን።

ጥያቄ፦
{question_am}

A. {option_a_am}
B. {option_b_am}
C. {option_c_am}
D. {option_d_am}
```

Use the equivalent instructions for the **English condition**:

```text
Read the following question and its four options carefully.

1. Select exactly one option.
2. On the first line, write only the letter of your selected
   option: A, B, C, or D.
3. Then explain why your selected answer is correct in 2–4
   sentences in English. Give a reason for your selection
   rather than merely repeating the answer.

Question:
{question_en}

A. {option_a_en}
B. {option_b_en}
C. {option_c_en}
D. {option_d_en}
```

**Send the instructions and question together as one user message in a fresh conversation for every item.** Use the same template throughout, and review the Amharic wording yourself before fixing it for the main evaluation.