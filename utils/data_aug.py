from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    BorderlineSMOTE,
    ADASYN
)

from imblearn.under_sampling import (
    RandomUnderSampler
)

from imblearn.combine import (
    SMOTEENN,
    SMOTETomek
)

def get_augmentation_methods(random_state=0):

    methods = {
        "No Augmentation": None,

        # Oversampling simple
        "RandomOverSampler": RandomOverSampler(random_state=random_state),

        # SMOTE classiques
        "SMOTE": SMOTE(random_state=random_state),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=random_state),
        "ADASYN": ADASYN(random_state=random_state),

        # Undersampling simple
        "RandomUnderSampler": RandomUnderSampler(random_state=random_state),

        # Méthodes hybrides
        "SMOTEENN": SMOTEENN(random_state=random_state),
        "SMOTETomek": SMOTETomek(random_state=random_state),

    }

    return methods

