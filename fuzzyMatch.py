import jellyfish as jf
import pandas as pd
import fuzzy as fz
import unicodedata
import itertools
import re

#performs fuzzy string matching based on a combination of edit distance and phonetic soudns
class fuzzyMatch:

    def __init__(self,fromVec):

        self.fromVec = fromVec

    #performs fuzzy string matching based on a combination of edit distance and phonetic soudns
    
    def matchNames(self,toVec,minMatchRatio=0.885):
        
        #creates 4 different kind of sound vectors
        def transcribePhonetic(vec): 
            DM = fz.DMetaphone(20)
            SX = fz.Soundex(4)
            if type(vec)!=pd.core.series.Series:
                vec = pd.Series(vec)
            vec = vec.map(lambda x: re.sub('\s+',' ',
                                           re.sub(r"[^\w'\s-]", '',
                                                  unicodedata.normalize('NFKD', unicode(x)).encode('ascii', 'ignore')))
                                                    .strip().replace(' - ','-'))
            df = vec.map(lambda x: [x]+DM(unicode(x))+[fz.nysiis(unicode(x))]+[jf.match_rating_codex(unicode(x))]+[jf.soundex(unicode(x))]).apply(pd.Series)
            df.loc[:,0] = df.loc[:,0].map(lambda x: x.lower().replace('-',' ').replace("'",""))
            return df
        
        #remove matches from both matchee (frDF) and matcher (toDF) lists
        def removeMatches(frDF,toDF,mtchDF):
            frDF = frDF.loc[~frDF.loc[:,6].isin(mtchDF.loc[:,'Original']),:]
            toDF = toDF.loc[~toDF.loc[:,6].isin(mtchDF.loc[:,'Match']),:]
            return frDF,toDF
        
        #Only choose matches that 
        def filterByEditDist(mrgDF,minMatchRatio=0.885):
            if mrgDF.empty:
                return mrgDF
            mrgDF['Dist'] = mrgDF.apply(lambda x: jf.jaro_winkler(unicode(x[0]),unicode(x[1])),axis=1)
            mrgDF = mrgDF.groupby('Original').apply(lambda x: x[x.Dist==x.Dist.max()]).reset_index(drop=True)
            return mrgDF.loc[mrgDF['Dist']>=minMatchRatio,['Original','Match']]
        
        #Find all matching rows between 2 dataframes where only k of N columns match
        def mergeOnKofNCols(frDF,toDF,colNames,k):
            colCmbns = [[1]+list(x) if x[0]==2 else list(x) for x in list(itertools.combinations(colNames,k))]
            mrgDF = pd.DataFrame(columns=['Original','Match'])
            for x in colCmbns:
                mrgDF = pd.concat([mrgDF,pd.merge(frDF,toDF,on=list(x),how='inner').loc[:,['6_x','6_y']].rename(columns={'6_x':'Original','6_y':'Match'})])
            return mrgDF.drop_duplicates()

        #if inputs empty
        frVec = self.fromVec
        if len(frVec)==0 or len(toVec)==0:
            return pd.DataFrame(columns=['Original','Match'])
        
        #remove duplicates
        frVec = frVec.drop_duplicates()
        toVec = toVec.drop_duplicates()

        #Transcribe names phonetically 
        frDF = transcribePhonetic(frVec)
        toDF = transcribePhonetic(toVec)

        #Concatenate with deduplicated list
        frDF = pd.concat([frDF.reset_index(drop=True),frVec.reset_index(drop=True)],axis=1,ignore_index=True)
        toDF = pd.concat([toDF.reset_index(drop=True),toVec.reset_index(drop=True)],axis=1,ignore_index=True)
        
        #first match directly on ASCII
        mtchDF = pd.merge(frDF,toDF,on=0,how='inner').loc[:,['6_x','6_y']].rename(columns={'6_x':'Original','6_y':'Match'})
        frDF, toDF = removeMatches(frDF,toDF,mtchDF)
        if frDF.empty or toDF.empty:
            return mtchDF

        #second match on all phonetic sounds
        phoneCols = [1,2,3,4,5]
        mrgDF = pd.merge(frDF.loc[:,1:6],toDF.loc[:,1:6],on=phoneCols,how='inner').iloc[:,[-1,-2]].rename(columns={'6_x':'Original','6_y':'Match'})
        mtchDF = pd.concat([mtchDF,mrgDF])
        frDF, toDF = removeMatches(frDF,toDF,mtchDF)
        if frDF.empty or toDF.empty:
            return mtchDF

        #third match on any k of n phonetic sounds in descending order of k
        for k in reversed(range(1,len(phoneCols)-1)):
            mrgDF = mergeOnKofNCols(frDF,toDF,phoneCols,k)
            #use edit distance to break ties
            mrgDF = filterByEditDist(mrgDF,minMatchRatio)
            mtchDF = pd.concat([mtchDF,mrgDF])
            frDF, toDF = removeMatches(frDF,toDF,mtchDF)
            if frDF.empty or toDF.empty:
                return mtchDF

        #Finally match by edit distance only
        mrgDF = pd.merge(frDF.assign(key=1),toDF.assign(key=1),on='key',how='outer').loc[:,['6_x','6_y']].rename(columns={'6_x':'Original','6_y':'Match'})

        #use edit distance to break ties
        mrgDF = filterByEditDist(mrgDF,minMatchRatio)
        mtchDF = pd.concat([mtchDF,mrgDF])
        frDF, toDF = removeMatches(frDF,toDF,mtchDF)

        self.matches = mtchDF

        return mtchDF

    #Use matchNames to replace items in vector with its corresponding match in another vector or dataframe
    def rplcNames(self,mtchDF=[],enforceMatch=True,minMatchRatio=0.885):

        frVec = self.fromVec
        if len(mtchDF)==0:
            mtchDF = self.matches

        if not isinstance(mtchDF, pd.DataFrame):
            toVec = mtchDF
            mtchDF = self.matchNames(toVec,minMatchRatio)
            
        frVec = pd.Series(frVec)
        rplcVec = frVec.map(mtchDF.set_index('Original')['Match'])
        
        #enforceMatch forces all items in frVec to be matched with toVec else error
        if not enforceMatch:
            rplcVec.loc[pd.isnull(rplcVec)] = frVec.loc[pd.isnull(rplcVec)]
        else:
            if sum(pd.isnull(rplcVec)*1)>0:
                return 'Error, Not everything matched!'
            
        return rplcVec

    #Performs nested match where matches must be within grouped subsets (e.g. markets must match within adm1 and adm2 districts)
    @staticmethod
    def grpByMatch(dfSub,dfRplc,cols):

        #First find and replace the first column
        findVec = dfSub.loc[:,cols[0]].drop_duplicates().reset_index(drop=True)
        rplcVec = dfRplc.loc[:,cols[0]].drop_duplicates().reset_index(drop=True)
        toSubVec = fuzzyMatch(findVec).matchNames(rplcVec).set_index('Original')['Match']
        dfSub.loc[:,cols[0]] = dfSub.loc[:,cols[0]].replace(toSubVec.to_dict())
        
        if len(cols)>1:
            #Now progressively increase the set of group by columns
            for i in range(1,len(cols)):
                grpByCols = cols[0:i]
                mtchCol = cols[i]
                
                #create groups from group by cols
                grouped = dfSub.loc[:,grpByCols+[mtchCol]].groupby(grpByCols)
                for name, group in grouped:
                    
                    #vector of names to be replaced
                    findVec = group.loc[:,mtchCol].drop_duplicates().dropna().reset_index(drop=True)
                    #vector of substitute names to be matched to
                    rplcDf  = pd.merge(dfRplc.loc[:,grpByCols+[mtchCol]].drop_duplicates(),group.loc[:,grpByCols].drop_duplicates(),on=grpByCols)
                    rplcVec = rplcDf.loc[:,mtchCol].drop_duplicates().dropna().reset_index(drop=True)
                    
                    #perform fuzzy match
                    if not findVec.empty and not rplcVec.empty:
                        toSubVec = fuzzyMatch(findVec).matchNames(rplcVec).set_index('Original')['Match']
                        
                        #substitute results into original DF
                        if not toSubVec.empty:
                            dfSub.loc[group.index.values,mtchCol] = dfSub.loc[group.index.values,mtchCol].replace(toSubVec.to_dict())

        return dfSub
