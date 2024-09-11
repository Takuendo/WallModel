#include "wallShearStressBCFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "Time.H"
#include "IFstream.H"
#include "fvPatchField.H"
#include "fixedGradientFvPatchField.H"

namespace Foam
{

// コンストラクタでのshear_フィールドの初期化
wallShearStressBCFvPatchVectorField::wallShearStressBCFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedGradientFvPatchVectorField(p, iF),
    shear_(p.size(), vector::zero)
{}


wallShearStressBCFvPatchVectorField::wallShearStressBCFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedGradientFvPatchVectorField(p, iF, dict),
    shear_(dict.lookupOrDefault("shear", vectorField(p.size(), vector::zero)))
{}


wallShearStressBCFvPatchVectorField::wallShearStressBCFvPatchVectorField
(
    const wallShearStressBCFvPatchVectorField& wbcvf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedGradientFvPatchVectorField(wbcvf, p, iF, mapper),
    shear_(wbcvf.shear_)
{}


// gradientメソッドの実装（shear_フィールドを返す）
const Field<vector>& wallShearStressBCFvPatchVectorField::gradient() const
{
    return shear_;
}

// updateCoeffsメソッドの実装
void wallShearStressBCFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // 現在のタイムステップと対応するファイルのパスを取得
    const Time& runTime = db().time();
    fileName timeDir = runTime.timeName();
    fileName shearFileName = timeDir / "wallshear";

    // ファイルを読み込み、剪断応力の値を取得
    IFstream shearFile(shearFileName);

    if (!shearFile.good())
    {
        FatalErrorInFunction
            << "剪断応力ファイルを開けませんでした: " << shearFileName << endl
            << exit(FatalError);
    }

    // shear_フィールドに値をコピー
    forAll(shear_, faceI)
    {
        scalar streamwise, vertical, spanwise;
        shearFile >> streamwise >> vertical >> spanwise;
        shear_[faceI] = vector(streamwise, vertical, spanwise);
    }

    // デバッグログ出力 (必要最低限の情報に絞る)
    Info << "Time = " << runTime.value() << ": Shear stress updated from " << shearFileName << endl;

    // 親クラスのupdateCoeffsを呼び出して剪断応力を適用
    fixedGradientFvPatchVectorField::updateCoeffs();
}

void wallShearStressBCFvPatchVectorField::write(Ostream& os) const
{
    // 親クラスのwrite()を呼び出し
    fixedGradientFvPatchVectorField::write(os);

    // shear_フィールドの書き込み
    os.writeKeyword("shear") << shear_ << token::END_STATEMENT << nl;
}

addToRunTimeSelectionTable
(
    fixedGradientFvPatchVectorField, // クラスのタイプにテンプレート引数を追加
    wallShearStressBCFvPatchVectorField, // クラス名
    dictionary // 引数のタイプ
);

} // End namespace Foam