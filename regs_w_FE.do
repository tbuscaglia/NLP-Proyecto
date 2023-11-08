clear
set more off
gl main "/Users/julianandelsman/Desktop/NLP/Final project/Data"
gl output "$main/Regresiones"
import delimited "$main/Regresion.csv", delimiter(",") clear

replace percentagechangeincompoundscore = subinstr(percentagechangeincompoundscore, ",", ".", .)
destring percentagechangeincompoundscore, replace

replace gapscore = subinstr(gapscore, ",", ".", .)
destring gapscore, replace

replace pmpositivecomments = subinstr(pmpositivecomments, ",", ".", .)
destring pmpositivecomments, replace

replace pmnegativecomments = subinstr(pmnegativecomments, ",", ".", .)
destring pmnegativecomments, replace

replace changeinpositivecomments = subinstr(changeinpositivecomments, ",", ".", .)
destring changeinpositivecomments, replace

replace changeinnegativecomments = subinstr(changeinnegativecomments, ",", ".", .)
destring changeinnegativecomments, replace

egen team_id = group(team)

* Regresions:

label var percentagechangeincompoundscore "Δ% in average compound score"
set more off
eststo clear
eststo: reg percentagechangeincompoundscore gapscore i.team_id, robust cluster(team_id)
eststo: reg pmpositivecomments gapscore i.team_id, robust cluster(team_id)
eststo: reg changeinpositivecomments gapscore i.team_id, robust cluster(team_id)
eststo: reg changeinnegativecomments gapscore i.team_id, robust cluster(team_id)
eststo: reg pmnegativecomments gapscore i.team_id, robust cluster(team_id)

esttab using "$output/RegsNlp.rtf", p se replace label noobs ///
keep(gapscore, relax) ///
cells(b(fmt(3) star) se(par fmt(3))) ///
stats(N, fmt(0 0 3) labels("Observations")) ///
addnotes ("Notes: Each column represents a separate regression. The unit of observation is each team-match combination. Robust standard errors are clustered at a team level.  *Significant at the 10% level. **Significant at the 5% level. ***Significant at the 1% level.")

*Frequency
histogram gapscore, bin(20) normal freq title("Gap Score Frequency")
graph export "$output/Gapscore.png", replace
