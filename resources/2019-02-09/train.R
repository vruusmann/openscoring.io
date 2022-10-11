audit = read.csv("audit.csv")
audit$Adjusted = as.factor(audit$Adjusted)

audit.formula = as.formula(
	Adjusted
	~
	# Include all raw columns as a starting point
	.
	# Append interactions
	+ Gender:Marital + Gender:Hours
	# Append the estimated hourly wage
	+ I(Income / (Hours * 52))
	# Take out the raw "Age" column, and append a binned one
	- Age + base::cut(Age, breaks = c(0, 18, 65, 120))
	# Take out the raw "Employment" column, and append a re-mapped one
	- Employment + plyr::mapvalues(Employment, c("PSFederal", "PSState", "PSLocal"), c("Public", "Public", "Public"))
)

audit.glm = glm(audit.formula, data = audit, family = "binomial")

library("dplyr")
library("r2pmml")

audit_sample = sample_n(audit, 10)
audit_sample$Adjusted = NULL

audit.glm = verify(audit.glm, newdata = audit_sample)

r2pmml(audit.glm, "LogisticRegressionAudit.pmml")
