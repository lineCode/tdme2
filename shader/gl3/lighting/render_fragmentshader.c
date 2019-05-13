// Based on: https://github.com/KhronosGroup/glTF-Sample-Viewer

#version 330

#define FALSE		0
#define MAX_LIGHTS	8
#define GAMMA		2.0
#define M_PI		3.141592653589793

struct Material {
	vec4 ambient;
	vec4 diffuse;
	vec4 specular;
	vec4 emission;
	float shininess;
};

struct Light {
	int enabled;
	vec4 ambient;
	vec4 diffuse;
	vec4 specular;
	vec4 position;
	vec3 spotDirection;
	float spotExponent;
	float spotCosCutoff;
	float constantAttenuation;
	float linearAttenuation;
	float quadraticAttenuation;
};

uniform mat4 cameraMatrix;
uniform Material material;
uniform Light lights[MAX_LIGHTS];
uniform vec3 cameraPosition;

#if defined(HAVE_SOLID_SHADING)
#else
	uniform sampler2D specularTextureUnit;
	uniform int specularTextureAvailable;

	uniform sampler2D normalTextureUnit;
	uniform int normalTextureAvailable;

	// material shininess
	float materialShininess;
#endif

// passed from geometry shader
in vec2 gsFragTextureUV;
in vec3 gsNormal;
in vec3 gsPosition;
in vec3 gsTangent;
in vec3 gsBitangent;
in vec4 gsEffectColorMul;
in vec4 gsEffectColorAdd;

// out
out vec4 outColor;

vec4 fragColor;

{$DEFINITIONS}

#if defined(HAVE_DEPTH_FOG)
	#define FOG_DISTANCE_NEAR			100.0
	#define FOG_DISTANCE_MAX				250.0
	#define FOG_RED						(255.0 / 255.0)
	#define FOG_GREEN					(255.0 / 255.0)
	#define FOG_BLUE						(255.0 / 255.0)
	in float fragDepth;
#endif

#if defined(HAVE_TERRAIN_SHADER)
	#define TERRAIN_LEVEL_0				-4.0
	#define TERRAIN_LEVEL_1				10.0
	#define TERRAIN_HEIGHT_BLEND		4.0
	#define TERRAIN_SLOPE_BLEND			10.0

	in float height;
	in float slope;
	uniform sampler2D grasTextureUnit;
	uniform sampler2D dirtTextureUnit;
	uniform sampler2D stoneTextureUnit;
	uniform sampler2D snowTextureUnit;
#else
	uniform sampler2D diffuseTextureUnit;
	uniform int diffuseTextureAvailable;
	uniform int diffuseTextureMaskedTransparency;
	uniform float diffuseTextureMaskedTransparencyThreshold;
#endif

#if defined(HAVE_SOLID_SHADING)
#else
	struct PBRMaterialInfo
	{
		float perceptualRoughness;    // roughness value, as authored by the model creator (input to shader)
		vec3 reflectance0;            // full reflectance color (normal incidence angle)

		float alphaRoughness;         // roughness mapped to a more linear change in the roughness (proposed by [2])
		vec3 diffuseColor;            // color contribution from diffuse lighting

		vec3 reflectance90;           // reflectance color at grazing angle
		vec3 specularColor;           // color contribution from specular lighting
	};

	struct PBRAngularInfo
	{
		float NdotL;                  // cos angle between normal and light direction
		float NdotV;                  // cos angle between normal and view direction
		float NdotH;                  // cos angle between normal and half vector
		float LdotH;                  // cos angle between light direction and half vector
		float VdotH;                  // cos angle between view direction and half vector
		vec3 padding;
	};

	// sRGB to linear approximation
	// see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
	vec4 SRGBtoLINEAR(vec4 srgbIn)
	{
	    return vec4(pow(srgbIn.xyz, vec3(GAMMA)), srgbIn.w);
	}

	// Lambert lighting
	// see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
	vec3 diffuse(PBRMaterialInfo materialInfo)
	{
	    return materialInfo.diffuseColor / M_PI;
	}

	PBRAngularInfo getAngularInfo(vec3 pointToLight, vec3 normal, vec3 view)
	{
	    // Standard one-letter names
	    vec3 n = normalize(normal);           // Outward direction of surface point
	    vec3 v = normalize(view);             // Direction from surface point to view
	    vec3 l = normalize(pointToLight);     // Direction from surface point to light
	    vec3 h = normalize(l + v);            // Direction of the vector between l and v

	    float NdotL = clamp(dot(n, l), 0.0, 1.0);
	    float NdotV = clamp(dot(n, v), 0.0, 1.0);
	    float NdotH = clamp(dot(n, h), 0.0, 1.0);
	    float LdotH = clamp(dot(l, h), 0.0, 1.0);
	    float VdotH = clamp(dot(v, h), 0.0, 1.0);

	    return PBRAngularInfo(
	        NdotL,
	        NdotV,
	        NdotH,
	        LdotH,
	        VdotH,
	        vec3(0, 0, 0)
	    );
	}

	// The following equation models the Fresnel reflectance term of the spec equation (aka F())
	// Implementation of fresnel from [4], Equation 15
	vec3 specularReflection(PBRMaterialInfo materialInfo, PBRAngularInfo angularInfo)
	{
	    return materialInfo.reflectance0 + (materialInfo.reflectance90 - materialInfo.reflectance0) * pow(clamp(1.0 - angularInfo.VdotH, 0.0, 1.0), 5.0);
	}

	// Smith Joint GGX
	// Note: Vis = G / (4 * NdotL * NdotV)
	// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3
	// see Real-Time Rendering. Page 331 to 336.
	// see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
	float visibilityOcclusion(PBRMaterialInfo materialInfo, PBRAngularInfo angularInfo)
	{
	    float NdotL = angularInfo.NdotL;
	    float NdotV = angularInfo.NdotV;
	    float alphaRoughnessSq = materialInfo.alphaRoughness * materialInfo.alphaRoughness;

	    float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
	    float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);

	    float GGX = GGXV + GGXL;
	    if (GGX > 0.0)
	    {
	        return 0.5 / GGX;
	    }
	    return 0.0;
	}

	// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
	// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
	// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
	float microfacetDistribution(PBRMaterialInfo materialInfo, PBRAngularInfo angularInfo)
	{
	    float alphaRoughnessSq = materialInfo.alphaRoughness * materialInfo.alphaRoughness;
	    float f = (angularInfo.NdotH * alphaRoughnessSq - angularInfo.NdotH) * angularInfo.NdotH + 1.0;
	    return alphaRoughnessSq / (M_PI * f * f);
	}

	vec3 getPointShade(vec3 pointToLight, PBRMaterialInfo materialInfo, vec3 normal, vec3 view)
	{
		PBRAngularInfo angularInfo = getAngularInfo(pointToLight, normal, view);

		if (angularInfo.NdotL > 0.0 || angularInfo.NdotV > 0.0)
		{
			// Calculate the shading terms for the microfacet specular shading model
			vec3 F = specularReflection(materialInfo, angularInfo);
			float Vis = visibilityOcclusion(materialInfo, angularInfo);
			float D = microfacetDistribution(materialInfo, angularInfo);

			// Calculation of analytical lighting contribution
			vec3 diffuseContrib = (1.0 - F) * diffuse(materialInfo);
			vec3 specContrib = F * Vis * D;

			// Obtain final intensity as reflectance (BRDF) scaled by the energy of the light (cosine law)
			return angularInfo.NdotL * (diffuseContrib + specContrib);
		}

		return vec3(0.0, 0.0, 0.0);
	}

	void applyDirectionalLight(Light light, PBRMaterialInfo materialInfo, vec3 normal, vec3 view, vec3 position)
	{
	    vec3 shade = getPointShade(-light.spotDirection, materialInfo, normal, view);
	    fragColor+= light.diffuse * vec4(shade, 1.0);
	}

//	vec3 applyPointLight(Light light, PBRMaterialInfo materialInfo, vec3 normal, vec3 view)
//	{
//	    vec3 pointToLight = light.position - v_Position;
//	    float distance = length(pointToLight);
//	    float attenuation = getRangeAttenuation(light.range, distance);
//	    vec3 shade = getPointShade(pointToLight, materialInfo, normal, view);
//	    return attenuation * light.intensity * light.color * shade;
//	}
//
//	vec3 applySpotLight(Light light, PBRMaterialInfo materialInfo, vec3 normal, vec3 view)
//	{
//	    vec3 pointToLight = light.position - v_Position;
//	    float distance = length(pointToLight);
//	    float rangeAttenuation = getRangeAttenuation(light.range, distance);
//	    float spotAttenuation = getSpotAttenuation(pointToLight, light.direction, light.outerConeCos, light.innerConeCos);
//	    vec3 shade = getPointShade(pointToLight, materialInfo, normal, view);
//	    return rangeAttenuation * spotAttenuation * light.intensity * light.color * shade;
//	}

	void computeDiffuseLights(in vec3 normal, in vec3 position, PBRMaterialInfo materialInfo) {
	    //
		vec3 view = normalize(cameraPosition - position);

		// process each light
		for (int i = 0; i < MAX_LIGHTS; i++) {
			Light light = lights[i];

			// skip on disabled lights
			if (light.enabled == FALSE) continue;

			//
			applyDirectionalLight(light, materialInfo, normal, view, position);
		}
	}

	void computeAmbientLights(int diffuseTextureAvailable, vec4 diffuseTextureColor) {
		// process each light
		for (int i = 0; i < MAX_LIGHTS; i++) {
			Light light = lights[i];

			// skip on disabled lights
			if (light.enabled == FALSE) continue;

			//
			fragColor = clamp(SRGBtoLINEAR(light.ambient) * SRGBtoLINEAR(material.ambient) * SRGBtoLINEAR((diffuseTextureAvailable == 1?diffuseTextureColor:vec4(1.0))), 0.0, 1.0);
		}
	}

	vec3 toneMap(vec3 color)
	{
		return pow(color, vec3(1.0 / GAMMA)) * 2.0;
	}

#endif

void main(void) {
	#if defined(HAVE_DEPTH_FOG)
		float fogStrength = 0.0;
		if (fragDepth > FOG_DISTANCE_NEAR) {
			fogStrength = (clamp(fragDepth, FOG_DISTANCE_NEAR, FOG_DISTANCE_MAX) - FOG_DISTANCE_NEAR) * 1.0 / (FOG_DISTANCE_MAX - FOG_DISTANCE_NEAR);
		}
	#endif

	// retrieve diffuse texture color value
	#if defined(HAVE_TERRAIN_SHADER)
		// no op
		int diffuseTextureAvailable = 1;
		vec4 diffuseTextureColor;
	#else
		vec4 diffuseTextureColor;
		if (diffuseTextureAvailable == 1) {
			// fetch from texture
			diffuseTextureColor = texture(diffuseTextureUnit, gsFragTextureUV);
			// check if to handle diffuse texture masked transparency
			if (diffuseTextureMaskedTransparency == 1) {
				// discard if beeing transparent
				if (diffuseTextureColor.a < diffuseTextureMaskedTransparencyThreshold) discard;
				// set to opqaue
				diffuseTextureColor.a = 1.0;
			}
		}
	#endif

	//
	fragColor = vec4(0.0, 0.0, 0.0, 0.0);
	fragColor+= clamp(material.emission, 0.0, 1.0);
	fragColor = fragColor * gsEffectColorMul;

	#if defined(HAVE_SOLID_SHADING)
		fragColor+= material.ambient;
		if (diffuseTextureAvailable == 1) {
			outColor = clamp((gsEffectColorAdd + diffuseTextureColor) * fragColor, 0.0, 1.0);
		} else {
			outColor = clamp(gsEffectColorAdd + fragColor, 0.0, 1.0);
		}
		if (outColor.a < 0.0001) discard;
	#else
		vec3 normal = gsNormal;

		// specular
		materialShininess = material.shininess;
		if (specularTextureAvailable == 1) {
			vec3 specularTextureValue = texture(specularTextureUnit, gsFragTextureUV).rgb;
			materialShininess =
				((0.33 * specularTextureValue.r) +
				(0.33 * specularTextureValue.g) +
				(0.33 * specularTextureValue.b)) * 255.0;
		}

		// compute normal
		if (normalTextureAvailable == 1) {
			vec3 normalVector = normalize(texture(normalTextureUnit, gsFragTextureUV).rgb * 2.0 - 1.0);
			normal = vec3(0.0, 0.0, 0.0);
			normal+= gsTangent * normalVector.x;
			normal+= gsBitangent * normalVector.y;
			normal+= gsNormal * normalVector.z;
		}

		// Metallic and Roughness material properties are packed together
		// In glTF, these factors can be specified by fixed scalar values
		// or from a metallic-roughness map
		float perceptualRoughness = 0.0;
		float metallic = 0.0;
		vec4 baseColor = vec4(0.0, 0.0, 0.0, 1.0);
		vec3 diffuseColor = vec3(0.0);
		vec3 specularColor = vec3(0.0);
		vec3 f0 = vec3(0.04);

		if (specularTextureAvailable == 1) {
			vec4 sgSample = SRGBtoLINEAR(texture(specularTextureUnit, gsFragTextureUV));
			perceptualRoughness = (1.0 - sgSample.a * material.shininess); // glossiness to roughness
			f0 = sgSample.rgb * SRGBtoLINEAR(material.specular).xyz; // specular
		} else {
			f0 = SRGBtoLINEAR(material.specular).rgb;
			perceptualRoughness = 1.0 - material.shininess / 127.0;
		}

		#if defined(HAVE_TERRAIN_SHADER)
			#if defined(HAVE_DEPTH_FOG)
				if (fogStrength < 1.0) {
			#else
				{
			#endif
					vec4 terrainBlending = vec4(0.0, 0.0, 0.0, 0.0); // gras, dirt, stone, snow
					if (height > TERRAIN_LEVEL_1) {
						float blendFactorHeight = clamp((height - TERRAIN_LEVEL_1) / TERRAIN_HEIGHT_BLEND, 0.0, 1.0);
						if (slope >= 45.0) {
							terrainBlending[2]+= blendFactorHeight; // stone
						} else
						if (slope >= 45.0 - TERRAIN_SLOPE_BLEND) {
							terrainBlending[2]+= blendFactorHeight * ((slope - (45.0 - TERRAIN_SLOPE_BLEND)) / TERRAIN_SLOPE_BLEND); // stone
							terrainBlending[3]+= blendFactorHeight * (1.0 - (slope - (45.0 - TERRAIN_SLOPE_BLEND)) / TERRAIN_SLOPE_BLEND); // snow
						} else {
							terrainBlending[3]+= blendFactorHeight; // snow
						}
					}
					if (height >= TERRAIN_LEVEL_0 && height < TERRAIN_LEVEL_1 + TERRAIN_HEIGHT_BLEND) {
						float blendFactorHeight = 1.0;
						if (height > TERRAIN_LEVEL_1) {
							blendFactorHeight = 1.0 - clamp((height - TERRAIN_LEVEL_1) / TERRAIN_HEIGHT_BLEND, 0.0, 1.0);
						} else
						if (height < TERRAIN_LEVEL_0 + TERRAIN_HEIGHT_BLEND) {
							blendFactorHeight = clamp((height - TERRAIN_LEVEL_0) / TERRAIN_HEIGHT_BLEND, 0.0, 1.0);
						}

						if (slope >= 45.0) {
							terrainBlending[2]+= blendFactorHeight; // stone
						} else
						if (slope >= 45.0 - TERRAIN_SLOPE_BLEND) {
							terrainBlending[2]+= blendFactorHeight * ((slope - (45.0 - TERRAIN_SLOPE_BLEND)) / TERRAIN_SLOPE_BLEND); // stone
							terrainBlending[1]+= blendFactorHeight * (1.0 - (slope - (45.0 - TERRAIN_SLOPE_BLEND)) / TERRAIN_SLOPE_BLEND); // dirt
						} else
						if (slope >= 26.0) {
							terrainBlending[1]+= blendFactorHeight; // dirt
						} else
						if (slope >= 26.0 - TERRAIN_SLOPE_BLEND) {
							terrainBlending[1]+= blendFactorHeight * ((slope - (26.0 - TERRAIN_SLOPE_BLEND)) / TERRAIN_SLOPE_BLEND); // dirt
							terrainBlending[0]+= blendFactorHeight * (1.0 - (slope - (26.0 - TERRAIN_SLOPE_BLEND)) / TERRAIN_SLOPE_BLEND); // gras
						} else {
							terrainBlending[0]+= blendFactorHeight; // gras
						}
					}
					if (height < TERRAIN_LEVEL_0 + TERRAIN_HEIGHT_BLEND) {
						float blendFactorHeight = 1.0;
						if (height > TERRAIN_LEVEL_0) {
							blendFactorHeight = 1.0 - clamp((height - TERRAIN_LEVEL_0) / TERRAIN_HEIGHT_BLEND, 0.0, 1.0);
						}
						// 0- meter
						terrainBlending[1]+= blendFactorHeight; // dirt
					}

					//
					diffuseTextureColor = gsEffectColorAdd;
					if (terrainBlending[0] > 0.001) diffuseTextureColor+= texture(grasTextureUnit, gsFragTextureUV) * terrainBlending[0];
					if (terrainBlending[1] > 0.001) diffuseTextureColor+= texture(dirtTextureUnit, gsFragTextureUV) * terrainBlending[1];
					if (terrainBlending[2] > 0.001) diffuseTextureColor+= texture(stoneTextureUnit, gsFragTextureUV) * terrainBlending[2];
					if (terrainBlending[3] > 0.001) diffuseTextureColor+= texture(snowTextureUnit, gsFragTextureUV) * terrainBlending[3];
					diffuseTextureColor = clamp(diffuseTextureColor, 0.0, 1.0);
			#if defined(HAVE_DEPTH_FOG)
				} else {
					diffuseTextureColor = gsEffectColorAdd + vec4(FOG_RED, FOG_GREEN, FOG_BLUE, 1.0);
				}
			#else
				}
			#endif
		#else
			if (diffuseTextureAvailable == 1) {
				baseColor = gsEffectColorAdd + SRGBtoLINEAR(diffuseTextureColor) * SRGBtoLINEAR(material.diffuse);
			} else {
				baseColor = gsEffectColorAdd + SRGBtoLINEAR(material.diffuse);
			}
		#endif

	    // f0 = specular
		specularColor = f0;
		float oneMinusSpecularStrength = 1.0 - max(max(f0.r, f0.g), f0.b);
		diffuseColor = baseColor.rgb * oneMinusSpecularStrength;

	    // TODO: roughness map

		diffuseColor = baseColor.rgb * (vec3(1.0) - f0) * (1.0 - metallic);

		specularColor = mix(f0, baseColor.rgb, metallic);

		perceptualRoughness = clamp(perceptualRoughness, 0.0, 1.0);
		metallic = clamp(metallic, 0.0, 1.0);

		// Roughness is authored as perceptual roughness; as is convention,
		// convert to material roughness by squaring the perceptual roughness [2].
		float alphaRoughness = perceptualRoughness * perceptualRoughness;

	    // Compute reflectance.
		float reflectance = max(max(specularColor.r, specularColor.g), specularColor.b);

	    vec3 specularEnvironmentR0 = specularColor.rgb;
	    // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
	    vec3 specularEnvironmentR90 = vec3(clamp(reflectance * 50.0, 0.0, 1.0));

		PBRMaterialInfo materialInfo = PBRMaterialInfo(
			perceptualRoughness,
			specularEnvironmentR0,
			alphaRoughness,
			diffuseColor,
			specularEnvironmentR90,
			specularColor
	    );

		// compute lights
		computeAmbientLights(diffuseTextureAvailable, diffuseTextureColor); // TODO: replace me with env ambient component
		computeDiffuseLights(normal, gsPosition, materialInfo);

		//
		// take effect colors into account
		fragColor.a = material.diffuse.a * gsEffectColorMul.a;
		outColor = vec4(toneMap(fragColor.rgb), baseColor.a);
	#endif

	#if defined(HAVE_BACK)
		gl_FragDepth = 1.0;
	#endif
	#if defined(HAVE_FRONT)
		gl_FragDepth = 0.0;
	#endif
	#if defined(HAVE_DEPTH_FOG)
		if (fogStrength > 0.0) {
			outColor = vec4(
				(outColor.rgb * (1.0 - fogStrength)) +
				vec3(FOG_RED, FOG_GREEN, FOG_BLUE) * fogStrength,
				1.0
			);
		}
	#endif
}
