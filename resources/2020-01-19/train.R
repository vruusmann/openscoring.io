library("dplyr")
library("plyr")
library("r2pmml")

audit.df = read.csv("audit.csv")
audit.df$Adjusted = as.factor(audit.df$Adjusted)

audit.terms = c("Adjusted ~ .")

# ?I
audit.terms = c(audit.terms, "+ I(log(Income)) - Income")

employment.newlevels = c(
	"Consultant" = "Private",
	"Private" = "Private",
	"PSFederal" = "Public",
	"PSLocal" = "Public",
	"PSState" = "Public",
	"SelfEmp" = "Private",
	"Volunteer" = "Other"
)

employment.newlevels.str = paste("c(", paste(lapply(names(employment.newlevels), function(x){ paste(shQuote(x), "=", shQuote(employment.newlevels[[x]])) }), collapse = ", "), ")", sep = "")
print(employment.newlevels.str)

# ?plyr::revalue
audit.terms = c(audit.terms, paste("+ plyr::revalue(Employment, replace = ", employment.newlevels.str, ") - Employment", sep = ""))

# ?interaction()
audit.terms = c(audit.terms, "+ Gender:Marital")

audit.formula = as.formula(paste(audit.terms, collapse = " "))
print(audit.formula)

audit.glm = glm(audit.formula, family = binomial(link = "logit"), data = audit.df)

audit.glm = r2pmml::verify(audit.glm, newdata = dplyr::sample_n(audit.df, 10))

r2pmml::r2pmml(audit.glm, "RExpAudit.pmml")
