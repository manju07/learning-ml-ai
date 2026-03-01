# Security Architecture: Guide for Architects

## Table of Contents
1. [Security Principles](#1-security-principles)
2. [Authentication](#2-authentication)
3. [Authorization (RBAC, ABAC)](#3-authorization-rbac-abac)
4. [OAuth 2.0 and OpenID Connect](#4-oauth-20-and-openid-connect)
5. [API Security](#5-api-security)
6. [Transport Security (TLS, mTLS)](#6-transport-security-tls-mtls)
7. [Secrets Management](#7-secrets-management)
8. [Zero Trust Architecture](#8-zero-trust-architecture)
9. [Security Hardening Checklist](#9-security-hardening-checklist)
10. [Practical Examples](#10-practical-examples)

---

## 1. Security Principles

### 1.1 Defense in Depth

Multiple layers: network, app, data, identity. No single point of failure.

### 1.2 Least Privilege

Grant minimum access required. Default deny.

### 1.3 Zero Trust

Never trust, always verify. No implicit trust based on network location.

### 1.4 Secure by Default

Encryption at rest and in transit; secure configs out of the box.

---

## 2. Authentication

### 2.1 Factors

- **Something you know**: Password, PIN
- **Something you have**: Token, phone
- **Something you are**: Biometric

### 2.2 Password Policies

- Minimum length (e.g., 12)
- Complexity (mixed case, numbers, symbols)
- No common passwords
- Hashing: bcrypt, Argon2, scrypt (never MD5/SHA1 for passwords)

```python
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
hashed = pwd_context.hash("secret")
pwd_context.verify("secret", hashed)
```

### 2.3 Multi-Factor Authentication (MFA)

TOTP (Google Authenticator), SMS (weaker), hardware keys (FIDO2).

### 2.4 JWT (JSON Web Token)

Stateless; contains claims. Signed (HMAC or RSA).

**Structure**: `header.payload.signature`

```python
import jwt
from datetime import datetime, timedelta

def create_token(user_id: str, secret: str, expires_min: int = 60):
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(minutes=expires_min),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, secret, algorithm="HS256")

def verify_token(token: str, secret: str):
    return jwt.decode(token, secret, algorithms=["HS256"])
```

**Best practices**:
- Short expiry (access: 15min; refresh: days)
- Use HTTPS only
- Store in httpOnly cookie or memory (not localStorage for XSS)
- Validate signature and expiry

---

## 3. Authorization (RBAC, ABAC)

### 3.1 RBAC (Role-Based Access Control)

Permissions assigned to roles; users get roles.

```
User -> Role -> Permission
Alice -> Admin -> *, *
Bob   -> Editor -> read, write
Carol -> Viewer -> read
```

### 3.2 ABAC (Attribute-Based Access Control)

Policy based on attributes: user, resource, environment.

```
Allow if user.role == "admin" OR (user.department == resource.owner AND resource.classification != "secret")
```

### 3.3 Policy Engines

- **Open Policy Agent (OPA)**: Rego language
- **Casbin**: Various models
- **AWS IAM**: Policy documents

### 3.4 Example: RBAC in Code

```python
from enum import Enum
from functools import wraps

class Role(Enum):
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"

PERMISSIONS = {
    Role.ADMIN: ["*"],
    Role.EDITOR: ["read", "write"],
    Role.VIEWER: ["read"],
}

def require_permission(permission: str):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, current_user=None, **kwargs):
            user_perms = PERMISSIONS.get(current_user.role, [])
            if "*" not in user_perms and permission not in user_perms:
                raise HTTPException(403, "Forbidden")
            return f(*args, current_user=current_user, **kwargs)
        return wrapper
    return decorator
```

---

## 4. OAuth 2.0 and OpenID Connect

### 4.1 OAuth 2.0 Roles

- **Resource Owner**: User
- **Client**: Application requesting access
- **Authorization Server**: Issues tokens
- **Resource Server**: API with protected resources

### 4.2 Grant Types

| Grant | Use Case |
|-------|----------|
| **Authorization Code** | Web apps, most secure |
| **PKCE** | SPAs, mobile (no client secret) |
| **Client Credentials** | Service-to-service |
| **Resource Owner Password** | Legacy, avoid |
| **Refresh Token** | Get new access token |

### 4.3 OAuth Flow (Authorization Code + PKCE)

```
1. Client generates code_verifier, code_challenge
2. Redirect user to Auth Server: /authorize?client_id=...&redirect_uri=...&code_challenge=...&state=...
3. User logs in, consents
4. Auth Server redirects: redirect_uri?code=...&state=...
5. Client exchanges: POST /token with code, code_verifier
6. Auth Server returns access_token, refresh_token
```

### 4.4 OpenID Connect (OIDC)

OAuth 2.0 + identity. Adds `id_token` (JWT) with user claims.

```
id_token: { sub, email, name, iss, aud, exp, ... }
```

### 4.5 Example: OAuth Client (Python)

```python
from authlib.integrations.httpx_client import AsyncOAuth2Client

client = AsyncOAuth2Client(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri="https://myapp.com/callback",
    scope="openid profile email",
)

# Step 1: Get auth URL
auth_url = client.create_authorization_url(
    "https://auth.example.com/authorize",
    state="random_state",
    code_challenge=generate_pkce_challenge(),
    code_challenge_method="S256",
)

# Step 2: After callback with code
token = await client.fetch_token(
    "https://auth.example.com/token",
    code=request.query_params["code"],
    code_verifier=stored_code_verifier,
)
# token: access_token, refresh_token, id_token, expires_in
```

---

## 5. API Security

### 5.1 Authentication

- **API Key**: Header `X-API-Key` — Simple; rotate regularly
- **Bearer Token**: `Authorization: Bearer <token>`
- **mTLS**: Client certificate

### 5.2 Input Validation

- Validate all inputs
- Parameterized queries (SQL injection)
- Sanitize output (XSS)
- Limit payload size

### 5.3 Rate Limiting

Prevent brute force, DoS. Per-IP, per-user, per-endpoint.

### 5.4 CORS

Restrict `Access-Control-Allow-Origin` to known domains.

### 5.5 Security Headers

```
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Content-Security-Policy: default-src 'self'
```

---

## 6. Transport Security (TLS, mTLS)

### 6.1 TLS

Encrypts data in transit. TLS 1.2 minimum; prefer 1.3.

### 6.2 mTLS (Mutual TLS)

Both client and server present certificates. Used for service-to-service.

```
Client cert -> Server validates
Server cert -> Client validates
```

### 6.3 Certificate Lifecycle

- Issuance (CA, Let's Encrypt)
- Rotation (short-lived, automation)
- Revocation (CRL, OCSP)

---

## 7. Secrets Management

### 7.1 Don'ts

- Hardcode in source
- Commit to git
- Log secrets
- Share via email/chat

### 7.2 Do's

- Use dedicated vault (HashiCorp Vault, AWS Secrets Manager)
- Rotate regularly
- Audit access
- Inject at runtime (env, mounted files)

### 7.3 Example: Vault Integration

```python
import hvac

client = hvac.Client(url='http://vault:8200', token=os.environ['VAULT_TOKEN'])
secret = client.secrets.kv.read_secret_version(path='prod/db')
db_password = secret['data']['data']['password']
```

### 7.4 Kubernetes Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
data:
  api-key: <base64>
```

Use external secret operators (e.g., External Secrets Operator) to sync from Vault/AWS.

---

## 8. Zero Trust Architecture

### 8.1 Principles

- Verify explicitly
- Least privilege
- Assume breach (segment, encrypt, monitor)

### 8.2 Implementation

- **Identity**: Strong auth (MFA), identity-aware proxy
- **Device**: Device compliance, posture
- **Network**: Micro-segmentation, no implicit trust
- **Data**: Encrypt at rest and in transit; DLP

### 8.3 Zero Trust Network Access (ZTNA)

Replace VPN with identity-based access. Per-app, not full network.

---

## 9. Security Hardening Checklist

### Application

- [ ] Input validation
- [ ] Parameterized queries
- [ ] Secure session management
- [ ] HTTPS only
- [ ] Security headers
- [ ] Dependency scanning (Dependabot, Snyk)

### Infrastructure

- [ ] Secrets in vault
- [ ] Least privilege IAM
- [ ] Network segmentation
- [ ] Encryption at rest

### Operations

- [ ] Logging and monitoring
- [ ] Incident response plan
- [ ] Security audits
- [ ] Penetration testing

---

## 10. Practical Examples

### 10.1 Spring Security with JWT Authentication

```java
@Configuration
@EnableWebSecurity
@EnableMethodSecurity(prePostEnabled = true)
public class SecurityConfig {
    
    private final JwtAuthenticationEntryPoint jwtAuthenticationEntryPoint;
    private final JwtAccessDeniedHandler jwtAccessDeniedHandler;
    private final JwtTokenProvider jwtTokenProvider;
    
    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder(12);
    }
    
    @Bean
    public AuthenticationManager authenticationManager(AuthenticationConfiguration config) throws Exception {
        return config.getAuthenticationManager();
    }
    
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        return http
            .csrf(csrf -> csrf.disable())
            .sessionManagement(session -> session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .exceptionHandling(exceptions -> exceptions
                .authenticationEntryPoint(jwtAuthenticationEntryPoint)
                .accessDeniedHandler(jwtAccessDeniedHandler))
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/auth/**", "/api/public/**", "/actuator/health").permitAll()
                .requestMatchers(HttpMethod.POST, "/api/users").hasRole("ADMIN")
                .requestMatchers(HttpMethod.DELETE, "/api/users/**").hasRole("ADMIN")
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                .anyRequest().authenticated())
            .oauth2ResourceServer(oauth2 -> oauth2
                .jwt(jwt -> jwt
                    .jwtAuthenticationConverter(jwtAuthenticationConverter())
                    .jwtDecoder(jwtDecoder())))
            .addFilterBefore(jwtAuthenticationFilter(), UsernamePasswordAuthenticationFilter.class)
            .build();
    }
    
    @Bean
    public JwtAuthenticationFilter jwtAuthenticationFilter() {
        return new JwtAuthenticationFilter(jwtTokenProvider);
    }
    
    @Bean
    public JwtDecoder jwtDecoder() {
        return NimbusJwtDecoder.withJwkSetUri("https://your-auth-server/.well-known/jwks.json").build();
    }
    
    @Bean
    public JwtAuthenticationConverter jwtAuthenticationConverter() {
        JwtGrantedAuthoritiesConverter authoritiesConverter = new JwtGrantedAuthoritiesConverter();
        authoritiesConverter.setAuthorityPrefix("ROLE_");
        authoritiesConverter.setAuthoritiesClaimName("roles");
        
        JwtAuthenticationConverter converter = new JwtAuthenticationConverter();
        converter.setJwtGrantedAuthoritiesConverter(authoritiesConverter);
        converter.setPrincipalClaimName("sub");
        return converter;
    }
}

// JWT Token Provider
@Component
@Slf4j
public class JwtTokenProvider {
    
    @Value("${app.jwt.secret}")
    private String jwtSecret;
    
    @Value("${app.jwt.expiration}")
    private int jwtExpirationInMs;
    
    @Value("${app.jwt.refresh-expiration}")
    private int refreshExpirationInMs;
    
    public String generateToken(UserPrincipal userPrincipal) {
        Date expiryDate = new Date(System.currentTimeMillis() + jwtExpirationInMs);
        
        return Jwts.builder()
            .setSubject(userPrincipal.getId())
            .setIssuedAt(new Date())
            .setExpiration(expiryDate)
            .claim("email", userPrincipal.getEmail())
            .claim("roles", userPrincipal.getAuthorities().stream()
                .map(GrantedAuthority::getAuthority)
                .collect(Collectors.toList()))
            .signWith(SignatureAlgorithm.HS512, jwtSecret)
            .compact();
    }
    
    public String generateRefreshToken() {
        Date expiryDate = new Date(System.currentTimeMillis() + refreshExpirationInMs);
        
        return Jwts.builder()
            .setIssuedAt(new Date())
            .setExpiration(expiryDate)
            .signWith(SignatureAlgorithm.HS512, jwtSecret)
            .compact();
    }
    
    public String getUserIdFromToken(String token) {
        Claims claims = Jwts.parser()
            .setSigningKey(jwtSecret)
            .parseClaimsJws(token)
            .getBody();
        return claims.getSubject();
    }
    
    public boolean validateToken(String authToken) {
        try {
            Jwts.parser().setSigningKey(jwtSecret).parseClaimsJws(authToken);
            return true;
        } catch (SignatureException ex) {
            log.error("Invalid JWT signature");
        } catch (MalformedJwtException ex) {
            log.error("Invalid JWT token");
        } catch (ExpiredJwtException ex) {
            log.error("Expired JWT token");
        } catch (UnsupportedJwtException ex) {
            log.error("Unsupported JWT token");
        } catch (IllegalArgumentException ex) {
            log.error("JWT claims string is empty");
        }
        return false;
    }
}

// JWT Authentication Filter
public class JwtAuthenticationFilter extends OncePerRequestFilter {
    
    private final JwtTokenProvider tokenProvider;
    private final CustomUserDetailsService userDetailsService;
    
    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, 
                                  FilterChain filterChain) throws ServletException, IOException {
        try {
            String jwt = getJwtFromRequest(request);
            
            if (StringUtils.hasText(jwt) && tokenProvider.validateToken(jwt)) {
                String userId = tokenProvider.getUserIdFromToken(jwt);
                UserDetails userDetails = userDetailsService.loadUserById(userId);
                
                UsernamePasswordAuthenticationToken authentication = 
                    new UsernamePasswordAuthenticationToken(userDetails, null, userDetails.getAuthorities());
                authentication.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                
                SecurityContextHolder.getContext().setAuthentication(authentication);
            }
        } catch (Exception ex) {
            log.error("Could not set user authentication in security context", ex);
        }
        
        filterChain.doFilter(request, response);
    }
    
    private String getJwtFromRequest(HttpServletRequest request) {
        String bearerToken = request.getHeader("Authorization");
        if (StringUtils.hasText(bearerToken) && bearerToken.startsWith("Bearer ")) {
            return bearerToken.substring(7);
        }
        return null;
    }
}
```

### 10.2 OAuth2 Resource Server with Spring Security

```java
@Configuration
@EnableWebSecurity
public class OAuth2ResourceServerConfig {
    
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        return http
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/public/**").permitAll()
                .requestMatchers("/api/user/**").hasAuthority("SCOPE_read")
                .requestMatchers(HttpMethod.POST, "/api/orders").hasAuthority("SCOPE_write")
                .anyRequest().authenticated())
            .oauth2ResourceServer(oauth2 -> oauth2
                .jwt(jwt -> jwt
                    .jwtAuthenticationConverter(jwtAuthenticationConverter())
                    .jwtDecoder(jwtDecoder())))
            .build();
    }
    
    @Bean
    public JwtDecoder jwtDecoder() {
        String jwkSetUri = "https://auth-server.example.com/.well-known/jwks.json";
        
        NimbusJwtDecoder jwtDecoder = NimbusJwtDecoder.withJwkSetUri(jwkSetUri)
            .jwsAlgorithm(SignatureAlgorithm.RS256)
            .cache(Duration.ofMinutes(5))
            .build();
            
        jwtDecoder.setJwtValidator(jwtValidator());
        return jwtDecoder;
    }
    
    @Bean
    public Converter<Jwt, AbstractAuthenticationToken> jwtAuthenticationConverter() {
        JwtGrantedAuthoritiesConverter authoritiesConverter = new JwtGrantedAuthoritiesConverter();
        authoritiesConverter.setAuthorityPrefix("");
        authoritiesConverter.setAuthoritiesClaimName("scope");
        
        JwtAuthenticationConverter converter = new JwtAuthenticationConverter();
        converter.setJwtGrantedAuthoritiesConverter(authoritiesConverter);
        return converter;
    }
    
    @Bean
    public OAuth2TokenValidator<Jwt> jwtValidator() {
        List<OAuth2TokenValidator<Jwt>> validators = new ArrayList<>();
        validators.add(new JwtTimestampValidator());
        validators.add(new JwtIssuerValidator("https://auth-server.example.com"));
        validators.add(audienceValidator());
        
        return new DelegatingOAuth2TokenValidator<>(validators);
    }
    
    @Bean
    public OAuth2TokenValidator<Jwt> audienceValidator() {
        return new JwtClaimValidator<List<String>>(JwtClaimNames.AUD, aud -> 
            aud != null && aud.contains("my-resource-server"));
    }
}

// OAuth2 Authorization Server Configuration
@Configuration
@EnableAuthorizationServer
public class AuthorizationServerConfig {
    
    @Bean
    @Order(1)
    public SecurityFilterChain authorizationServerSecurityFilterChain(HttpSecurity http) throws Exception {
        OAuth2AuthorizationServerConfiguration.applyDefaultSecurity(http);
        
        http.exceptionHandling(exceptions -> exceptions
            .defaultAuthenticationEntryPointFor(
                new LoginUrlAuthenticationEntryPoint("/login"),
                new MediaTypeRequestMatcher(MediaType.TEXT_HTML)))
            .oauth2ResourceServer(OAuth2ResourceServerConfigurer::jwt);
            
        return http.build();
    }
    
    @Bean
    public RegisteredClientRepository registeredClientRepository() {
        RegisteredClient webClient = RegisteredClient.withId("web-client")
            .clientId("web-app")
            .clientSecret("{noop}secret")
            .clientAuthenticationMethod(ClientAuthenticationMethod.CLIENT_SECRET_BASIC)
            .authorizationGrantType(AuthorizationGrantType.AUTHORIZATION_CODE)
            .authorizationGrantType(AuthorizationGrantType.REFRESH_TOKEN)
            .redirectUri("http://localhost:3000/callback")
            .scope(OidcScopes.OPENID)
            .scope("read")
            .scope("write")
            .clientSettings(ClientSettings.builder()
                .requireAuthorizationConsent(true)
                .requireProofKey(true)
                .build())
            .tokenSettings(TokenSettings.builder()
                .accessTokenTimeToLive(Duration.ofMinutes(15))
                .refreshTokenTimeToLive(Duration.ofHours(1))
                .reuseRefreshTokens(false)
                .build())
            .build();
            
        return new InMemoryRegisteredClientRepository(webClient);
    }
    
    @Bean
    public JWKSource<SecurityContext> jwkSource() {
        KeyPair keyPair = generateRsaKey();
        RSAPublicKey publicKey = (RSAPublicKey) keyPair.getPublic();
        RSAPrivateKey privateKey = (RSAPrivateKey) keyPair.getPrivate();
        
        RSAKey rsaKey = new RSAKey.Builder(publicKey)
            .privateKey(privateKey)
            .keyID(UUID.randomUUID().toString())
            .build();
            
        JWKSet jwkSet = new JWKSet(rsaKey);
        return new ImmutableJWKSet<>(jwkSet);
    }
    
    @Bean
    public JwtDecoder jwtDecoder(JWKSource<SecurityContext> jwkSource) {
        return OAuth2AuthorizationServerConfiguration.jwtDecoder(jwkSource);
    }
}
```

### 10.3 Method-Level Security with Annotations

```java
@RestController
@RequestMapping("/api/users")
@PreAuthorize("hasRole('USER')")
@Validated
public class UserController {
    
    private final UserService userService;
    
    @GetMapping("/{userId}")
    @PreAuthorize("hasRole('ADMIN') or #userId == authentication.principal.id")
    public ResponseEntity<UserDto> getUser(@PathVariable String userId) {
        UserDto user = userService.findById(userId);
        return ResponseEntity.ok(user);
    }
    
    @PutMapping("/{userId}")
    @PreAuthorize("hasRole('ADMIN') or (#userId == authentication.principal.id and hasPermission(#userDto, 'EDIT'))")
    public ResponseEntity<UserDto> updateUser(@PathVariable String userId, 
                                            @RequestBody @Valid UserDto userDto) {
        UserDto updatedUser = userService.update(userId, userDto);
        return ResponseEntity.ok(updatedUser);
    }
    
    @DeleteMapping("/{userId}")
    @PreAuthorize("hasRole('ADMIN')")
    @PostAuthorize("returnObject.statusCode.is2xxSuccessful()")
    public ResponseEntity<Void> deleteUser(@PathVariable String userId) {
        userService.delete(userId);
        return ResponseEntity.noContent().build();
    }
    
    @PostMapping("/{userId}/roles")
    @PreAuthorize("hasRole('ADMIN') and hasPermission(#userId, 'User', 'MANAGE_ROLES')")
    public ResponseEntity<Void> assignRole(@PathVariable String userId, 
                                         @RequestBody RoleAssignmentDto roleDto) {
        userService.assignRole(userId, roleDto.getRoleName());
        return ResponseEntity.ok().build();
    }
}

// Custom Permission Evaluator
@Component
public class CustomPermissionEvaluator implements PermissionEvaluator {
    
    private final UserService userService;
    private final RoleService roleService;
    
    @Override
    public boolean hasPermission(Authentication authentication, Object targetDomainObject, Object permission) {
        if (authentication == null || targetDomainObject == null || permission == null) {
            return false;
        }
        
        UserPrincipal principal = (UserPrincipal) authentication.getPrincipal();
        String permissionString = permission.toString();
        
        if (targetDomainObject instanceof UserDto) {
            return hasUserPermission(principal, (UserDto) targetDomainObject, permissionString);
        }
        
        return false;
    }
    
    @Override
    public boolean hasPermission(Authentication authentication, Serializable targetId, 
                               String targetType, Object permission) {
        if (authentication == null || targetId == null || targetType == null || permission == null) {
            return false;
        }
        
        UserPrincipal principal = (UserPrincipal) authentication.getPrincipal();
        String permissionString = permission.toString();
        
        switch (targetType) {
            case "User":
                return hasUserPermission(principal, targetId.toString(), permissionString);
            case "Order":
                return hasOrderPermission(principal, targetId.toString(), permissionString);
            default:
                return false;
        }
    }
    
    private boolean hasUserPermission(UserPrincipal principal, UserDto user, String permission) {
        // Admin can do anything
        if (principal.getAuthorities().stream()
            .anyMatch(auth -> auth.getAuthority().equals("ROLE_ADMIN"))) {
            return true;
        }
        
        // Users can edit their own profile
        if ("EDIT".equals(permission) && user.getId().equals(principal.getId())) {
            return true;
        }
        
        return false;
    }
    
    private boolean hasUserPermission(UserPrincipal principal, String userId, String permission) {
        if ("MANAGE_ROLES".equals(permission)) {
            return principal.getAuthorities().stream()
                .anyMatch(auth -> auth.getAuthority().equals("ROLE_ADMIN"));
        }
        
        return false;
    }
}
```

### 10.4 Advanced Rate Limiting with Spring Security

```java
@Component
@Slf4j
public class SecurityRateLimitingFilter extends OncePerRequestFilter {
    
    private final RateLimitService rateLimitService;
    private final ObjectMapper objectMapper;
    
    @Override
    protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, 
                                  FilterChain filterChain) throws ServletException, IOException {
        
        String clientIdentifier = extractClientIdentifier(request);
        String endpoint = request.getRequestURI();
        
        RateLimitResult result = rateLimitService.checkRateLimit(
            clientIdentifier, endpoint, getRateLimitConfig(request, endpoint));
            
        if (!result.isAllowed()) {
            handleRateLimitExceeded(response, result);
            return;
        }
        
        // Add rate limit headers
        response.setHeader("X-RateLimit-Limit", String.valueOf(result.getLimit()));
        response.setHeader("X-RateLimit-Remaining", String.valueOf(result.getRemaining()));
        response.setHeader("X-RateLimit-Reset", String.valueOf(result.getResetTime()));
        
        filterChain.doFilter(request, response);
    }
    
    private String extractClientIdentifier(HttpServletRequest request) {
        // Priority: User ID > API Key > IP Address
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null && auth.isAuthenticated() && auth.getPrincipal() instanceof UserPrincipal) {
            return "user:" + ((UserPrincipal) auth.getPrincipal()).getId();
        }
        
        String apiKey = request.getHeader("X-API-Key");
        if (StringUtils.hasText(apiKey)) {
            return "api:" + apiKey;
        }
        
        return "ip:" + getClientIpAddress(request);
    }
    
    private RateLimitConfig getRateLimitConfig(HttpServletRequest request, String endpoint) {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        
        // Different limits based on user role
        if (auth != null && auth.getAuthorities().stream()
            .anyMatch(a -> a.getAuthority().equals("ROLE_PREMIUM"))) {
            return RateLimitConfig.builder()
                .limit(1000)
                .window(Duration.ofHours(1))
                .build();
        }
        
        // Different limits for sensitive endpoints
        if (endpoint.startsWith("/api/admin/")) {
            return RateLimitConfig.builder()
                .limit(100)
                .window(Duration.ofHours(1))
                .build();
        }
        
        return RateLimitConfig.builder()
            .limit(100)
            .window(Duration.ofMinutes(15))
            .build();
    }
    
    private void handleRateLimitExceeded(HttpServletResponse response, RateLimitResult result) 
            throws IOException {
        response.setStatus(HttpStatus.TOO_MANY_REQUESTS.value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setHeader("Retry-After", String.valueOf(result.getRetryAfterSeconds()));
        
        Map<String, Object> errorResponse = Map.of(
            "error", "rate_limit_exceeded",
            "message", "Rate limit exceeded. Try again later.",
            "limit", result.getLimit(),
            "retryAfter", result.getRetryAfterSeconds()
        );
        
        response.getWriter().write(objectMapper.writeValueAsString(errorResponse));
    }
}

@Service
public class RateLimitService {
    
    private final StringRedisTemplate redisTemplate;
    
    public RateLimitResult checkRateLimit(String clientId, String endpoint, RateLimitConfig config) {
        String key = String.format("rate_limit:%s:%s", clientId, endpoint);
        long currentTime = System.currentTimeMillis();
        long windowStart = currentTime - config.getWindow().toMillis();
        
        String script = """
            local key = KEYS[1]
            local window_start = tonumber(ARGV[1])
            local current_time = tonumber(ARGV[2])
            local limit = tonumber(ARGV[3])
            local window_ms = tonumber(ARGV[4])
            
            -- Remove expired entries
            redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
            
            -- Count current requests
            local current_count = redis.call('ZCARD', key)
            
            if current_count < limit then
                -- Add current request
                redis.call('ZADD', key, current_time, current_time)
                redis.call('EXPIRE', key, math.ceil(window_ms / 1000))
                
                return {1, current_count + 1, limit - current_count - 1}
            else
                -- Rate limit exceeded
                local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
                local reset_time = 0
                if #oldest > 0 then
                    reset_time = oldest[2] + window_ms
                end
                
                return {0, current_count, 0, reset_time}
            end
        """;
        
        List<Object> result = redisTemplate.execute(
            new DefaultRedisScript<>(script, List.class),
            Collections.singletonList(key),
            String.valueOf(windowStart),
            String.valueOf(currentTime),
            String.valueOf(config.getLimit()),
            String.valueOf(config.getWindow().toMillis())
        );
        
        boolean allowed = ((Number) result.get(0)).intValue() == 1;
        int currentCount = ((Number) result.get(1)).intValue();
        int remaining = ((Number) result.get(2)).intValue();
        
        return RateLimitResult.builder()
            .allowed(allowed)
            .limit(config.getLimit())
            .remaining(remaining)
            .resetTime(allowed || result.size() < 4 ? 
                currentTime + config.getWindow().toMillis() : 
                ((Number) result.get(3)).longValue())
            .build();
    }
}
```

### 10.5 mTLS Configuration

```java
@Configuration
@ConditionalOnProperty(name = "app.security.mtls.enabled", havingValue = "true")
public class MutualTLSConfig {
    
    @Value("${app.security.mtls.truststore.location}")
    private String truststoreLocation;
    
    @Value("${app.security.mtls.truststore.password}")
    private String truststorePassword;
    
    @Bean
    public ServletWebServerFactory servletWebServerFactory() {
        TomcatServletWebServerFactory factory = new TomcatServletWebServerFactory();
        factory.addAdditionalTomcatConnectors(createMTLSConnector());
        return factory;
    }
    
    private Connector createMTLSConnector() {
        Connector connector = new Connector("org.apache.coyote.http11.Http11NioProtocol");
        connector.setPort(8443);
        connector.setSecure(true);
        
        Http11NioProtocol protocol = (Http11NioProtocol) connector.getProtocolHandler();
        protocol.setSSLEnabled(true);
        protocol.setKeystoreFile(new File("classpath:keystore.p12").getAbsolutePath());
        protocol.setKeystoreType("PKCS12");
        protocol.setKeystorePass("changeit");
        protocol.setTruststoreFile(truststoreLocation);
        protocol.setTruststorePass(truststorePassword);
        protocol.setClientAuth("true"); // Require client certificates
        
        return connector;
    }
    
    @Bean
    @Order(1)
    public SecurityFilterChain mtlsSecurityFilterChain(HttpSecurity http) throws Exception {
        return http
            .requestMatchers(requestMatchers -> requestMatchers
                .requestMatchers("/api/internal/**", "/api/b2b/**"))
            .x509(x509 -> x509
                .subjectPrincipalRegex("CN=(.*?)(?:,|$)")
                .userDetailsService(clientCertificateUserDetailsService()))
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/internal/**").hasRole("SERVICE")
                .requestMatchers("/api/b2b/**").hasRole("PARTNER")
                .anyRequest().authenticated())
            .build();
    }
    
    @Bean
    public UserDetailsService clientCertificateUserDetailsService() {
        return new ClientCertificateUserDetailsService();
    }
    
    @Component
    public static class ClientCertificateUserDetailsService implements UserDetailsService {
        
        private final Map<String, ServiceAccount> serviceAccounts;
        
        public ClientCertificateUserDetailsService() {
            // Load service accounts from configuration
            this.serviceAccounts = Map.of(
                "payment-service", ServiceAccount.builder()
                    .name("payment-service")
                    .roles(Set.of("ROLE_SERVICE", "ROLE_PAYMENT"))
                    .build(),
                "partner-api", ServiceAccount.builder()
                    .name("partner-api")
                    .roles(Set.of("ROLE_PARTNER"))
                    .build()
            );
        }
        
        @Override
        public UserDetails loadUserByUsername(String commonName) throws UsernameNotFoundException {
            ServiceAccount account = serviceAccounts.get(commonName);
            if (account == null) {
                throw new UsernameNotFoundException("Service account not found: " + commonName);
            }
            
            return User.builder()
                .username(account.getName())
                .password("") // No password for certificate auth
                .authorities(account.getRoles().toArray(new String[0]))
                .build();
        }
    }
}
```

---

## Summary

| Topic | Key Takeaway |
|-------|--------------|
| **Auth** | Strong passwords, MFA, short-lived tokens |
| **OAuth/OIDC** | Authorization Code + PKCE for apps |
| **Authorization** | RBAC common; ABAC for fine-grained |
| **API Security** | Validate input, rate limit, security headers |
| **Secrets** | Vault, rotate, never in code |
| **Zero Trust** | Verify always, least privilege, assume breach |

---

## Further Reading

- OAuth 2.0: https://oauth.net/2/
- OWASP Top 10: https://owasp.org/www-project-top-ten/
- Zero Trust: https://www.nist.gov/publications/zero-trust-architecture
