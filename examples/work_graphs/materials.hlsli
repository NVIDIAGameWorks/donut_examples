/*
* Copyright (c) 2014-2024, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

float3 EvaluateMaterial_Lambert(Material material, float3 worldPosition, float3 worldNormal, float3 lightToPointDir, float3 lightColor, float lightAttenuation)
{
	return saturate(dot(-lightToPointDir, worldNormal)) * lightColor * lightAttenuation * material.baseColor;
}

float3 EvaluateMaterial_Phong(Material material, float3 worldPosition, float3 worldNormal, float3 lightToPointDir, float3 lightColor, float lightAttenuation, float3 viewPosition)
{
	const float3 specularColor = material.param1;
	const float specularPower = material.param2;
	const float3 pointToEyeDir = normalize(viewPosition-worldPosition); // Starting from worldPosition, pointing towards camera position

	const float3 diffuse = EvaluateMaterial_Lambert(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation);
	const float3 specular = pow(saturate(dot(reflect(lightToPointDir, worldNormal), pointToEyeDir)), specularPower) * specularColor * lightColor;
	return diffuse + specular;
}

float3 EvaluateMaterial_Metallic(Material material, float3 worldPosition, float3 worldNormal, float3 lightToPointDir, float3 lightColor, float lightAttenuation, float3 viewPosition)
{
	const float specularPower = 60.0f;
	const float3 pointToEyeDir = normalize(viewPosition-worldPosition); // Starting from worldPosition, pointing towards camera position
	const float3 specular = pow(saturate(dot(reflect(lightToPointDir, worldNormal), pointToEyeDir)), specularPower) * material.baseColor * lightColor;
	return specular;
}

float3 EvaluateMaterial_Velvet(Material material, float3 worldPosition, float3 worldNormal, float3 lightToPointDir, float3 lightColor, float lightAttenuation, float3 viewPosition)
{
	const float roughness = material.param1.x;

	const float3 specularColor = sqrt(material.baseColor*0.25f);
	const float3 pointToEyeDir = normalize(viewPosition-worldPosition); // Starting from worldPosition, pointing towards camera position
	const float3 halfDir = normalize(pointToEyeDir-lightToPointDir);

	// Some dots
	const float cosNrmHalfDir = saturate(dot(worldNormal, halfDir));
	const float cosNrmLightDir = saturate(dot(-lightToPointDir, worldNormal));
	const float cosNrmViewDir = saturate(dot(pointToEyeDir, worldNormal));
	const float cosViewHalfDir = saturate(dot(pointToEyeDir, halfDir));

	float distribution;
	{
		const float roughnessSq = roughness*roughness;
		const float cosNrmHalfDirSq = cosNrmHalfDir*cosNrmHalfDir;
		const float sinNrmHalfDirSq = 1-cosNrmHalfDirSq;
		const float sinNrmHalfDirQd = sinNrmHalfDirSq*sinNrmHalfDirSq;
		distribution = saturate((sinNrmHalfDirQd + 4*exp(-cosNrmHalfDirSq/(sinNrmHalfDirSq*roughnessSq))) / (c_PI*(1+4*roughnessSq)*sinNrmHalfDirQd));
	}

	const float v = 1/(4*(cosNrmLightDir+cosNrmViewDir-cosNrmLightDir*cosNrmViewDir));
	const float3 fresnel = specularColor + (1-specularColor) * pow((1-cosViewHalfDir), 5);
	const float3 specular = fresnel * (distribution*v*c_PI*cosNrmLightDir) * lightColor;

	const float3 diffuse = EvaluateMaterial_Lambert(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation) * 0.25f;
	return diffuse + specular;
}

float3 EvaluateMaterial_Flakes(Material material, float3 worldPosition, float3 worldNormal, float3 lightToPointDir, float3 lightColor, float lightAttenuation, float3 viewPosition)
{
	const float3 specularColor = material.param1;
	const float specularPower = material.param2;
	const float granularity = material.param3;
	const float diffusionStrength = 0.3f;
	const float3 pointToEyeDir = normalize(viewPosition-worldPosition); // Starting from worldPosition, pointing towards camera position

	const int3 flake = (int3)(worldPosition/granularity);
	const uint rand0 = Random(flake.x*flake.y+flake.z);
	const uint rand1 = Random(rand0);
	const uint rand2 = Random(rand1);
	const float3 offset = normalize(float3(rand0,rand1,rand2))*diffusionStrength;
	const float3 flakeWorldNormal = normalize(worldNormal+offset);

	const float3 diffuse = EvaluateMaterial_Lambert(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation);
	const float3 specular = pow(saturate(dot(reflect(lightToPointDir, worldNormal), pointToEyeDir)), specularPower) * specularColor * lightColor;
	const float3 specularFlakes = pow(saturate(dot(reflect(lightToPointDir, flakeWorldNormal), pointToEyeDir)), specularPower) * specularColor * lightColor;
	return diffuse + (specular+specularFlakes)*0.5f;
}

float3 EvaluateMaterial_Faceted(Material material, float3 worldPosition, float3 worldNormal, float3 lightToPointDir, float3 lightColor, float lightAttenuation, float3 viewPosition)
{
	lightToPointDir = -lightToPointDir; // Make all lights point towards the object

	return EvaluateMaterial_Metallic(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation, viewPosition);
}

float3 EvaluateMaterial_Stan(Material material, float3 worldPosition, float3 worldNormal, float3 lightToPointDir, float3 lightColor, float lightAttenuation)
{
	const float3 linesColor = material.param1;
	const float linesThickness = material.param2;
	const float linesSpacing = material.param3;

	const float3 patternPos = fmod(abs(worldPosition),(float3)(linesSpacing+linesThickness));
	material.baseColor = lerp(linesColor, material.baseColor, any((patternPos-linesThickness) < 0));

	return EvaluateMaterial_Lambert(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation);
}

float3 EvaluateMaterial_Checker(Material material, float3 worldPosition, float3 worldNormal, float3 lightToPointDir, float3 lightColor, float lightAttenuation, float3 viewPosition)
{
	const float3 baseColor2 = material.param1;
	const float checkerSize = material.param2;
	const float specularPower = material.param3;

	const float3 patternPos = fmod(abs(worldPosition),(float3)checkerSize*2);
	const bool alternate = any((patternPos.xy-checkerSize.xx) < 0) ^ ((patternPos.z-checkerSize) > 0);
	if (alternate)
	{
		material.baseColor = baseColor2;
		return EvaluateMaterial_Phong(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation, viewPosition);
	}
	else return EvaluateMaterial_Lambert(material, worldPosition, worldNormal, lightToPointDir, lightColor, lightAttenuation);
}