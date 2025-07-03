
ELEM1 = $(word $2,$(subst -, ,$1))
ELEM2 = $(word $2,$(subst ., ,$(subst -, ,$1)))
ELEM3 = $(word $2,$(subst /, ,$(subst -, ,$1)))
ELEM4 = $(word $2,$(subst _, ,$1))
ELEM5 = $(word $2,$(subst /, ,$1))
words_3_to_n = $(subst $(space),_,$(wordlist 3, $(words $(subst _, ,$1)), $(subst _, ,$1)))

define MVS_PATH
$(call ELEM5,$(1),1)/$(call ELEM5,$(1),2)/mvs/$(call ELEM5,$(1),4)
endef

define newline


endef

space := $(empty) $(empty)