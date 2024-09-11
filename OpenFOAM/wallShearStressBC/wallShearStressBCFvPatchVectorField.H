#ifndef wallShearStressBCFvPatchVectorField_H
#define wallShearStressBCFvPatchVectorField_H

#include "fvPatchFields.H"
#include "fvPatchField.H"
#include "fixedGradientFvPatchFields.H"
#include "vectorField.H"

namespace Foam
{

class wallShearStressBCFvPatchVectorField : public fixedGradientFvPatchVectorField
{
    // Private data
    vectorField shear_;

public:

    //- Construct from patch and internal field
    wallShearStressBCFvPatchVectorField
    (
        const fvPatch&,
        const DimensionedField<vector, volMesh>&
    );

    //- Construct from patch, internal field and dictionary
    wallShearStressBCFvPatchVectorField
    (
        const fvPatch&,
        const DimensionedField<vector, volMesh>&,
        const dictionary&
    );

    //- Construct by mapping given wallShearStressBCFvPatchVectorField
    wallShearStressBCFvPatchVectorField
    (
        const wallShearStressBCFvPatchVectorField&,
        const fvPatch&,
        const DimensionedField<vector, volMesh>&,
        const fvPatchFieldMapper&
    );

    //- Return the gradient
    virtual const Field<vector>& gradient() const;


    //- Update the coefficients associated with the patch field
    virtual void updateCoeffs();

    //- Write
    virtual void write(Ostream&) const;

    // typeName() メソッドの宣言 (修正)
    static const char* typeName_() { return "wallShearStressBC"; }
};


} // End namespace Foam

#endif