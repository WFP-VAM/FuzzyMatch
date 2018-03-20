# FuzzyMatch

Performs 'ensemble' fuzzy matching using 4 different phonetic transcription algorithms - DMetaphone(fuzzy), DSoundex(fuzzy), match rating codex (jellyfish), soundex(jellyfish). The function then sequentially matches in descending order of agreement with the above algorithms -- IE first 4/4 matches are replaced then 3/4 then 2/4 then 1/4 and finally matches are found be edit distance. The class can also be applied over two grouped dataframes of the same width where where matches must be nested within grouped subsets (e.g. markets must match within adm1 and adm2 districts)

##Example to call class

items2Match2 - destination vector containing items to match to  
items2BMatch - source vector containing items to be matched  

matchNames(items2BMatch).matchNames(items2Match2) - returns dataframe of Original and Match columns  
matchNames(items2BMatch).rplcNames(items2Match2) - returns vector of original length as items2BMatch with replacements where matched or else the original item  
fuzzyMatch.grpByMatch(dfitems2BMatch,dfitems2Match2) - returns dataframe of dimension of dfitems2BMatch with matches where found or else filled by original item  





