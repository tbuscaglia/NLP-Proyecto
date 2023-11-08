clear
set more off
gl main "/Users/julianandelsman/Desktop/NLP/Final project/Data"
gl output "$main/Regresiones"
import delimited "$main/Data_gap.csv", delimiter(",") clear

local variables "words achieve adj adverb affect affiliation anger anx article assent auxverb bio body cause certain cogproc compare conj death differ discrep drives family feel female filler focusfuture focuspast focuspresent friend funct health hear home i informal ingest insight interrog ipron leisure male money motion negate negemo netspeak nonflu number percept posemo power ppron prep pronoun quant relativ relig reward risk sad see sexual shehe social space swear tentat they time verb we work you"

foreach var of local variables{
replace `var' = subinstr(`var', ",", ".", .)
destring `var', replace
}

foreach var of local variables{
corr `var' gapscore
}
