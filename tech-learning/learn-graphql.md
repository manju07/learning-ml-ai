# GraphQL Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Intermediate Concepts](#intermediate-concepts)
4. [Advanced Concepts](#advanced-concepts)
5. [Server-Side Concepts](#server-side-concepts)
6. [Schema Design](#schema-design)
7. [Best Practices](#best-practices)
8. [Quick Reference](#quick-reference)

---

## Introduction

### What is GraphQL?

GraphQL is a **query language for APIs** and a **runtime for executing those queries**. Unlike REST APIs where you get fixed data structures, GraphQL lets you:

- **Request exactly what you need** - No more, no less
- **Get multiple resources in one request** - Reduce round trips
- **Strongly typed** - Know exactly what data structure you'll get
- **Self-documenting** - The schema describes what's available

### Key Benefits

✅ **Efficient Data Fetching** - Request only the fields you need  
✅ **Single Endpoint** - One endpoint handles all operations  
✅ **Type Safety** - Catch errors before runtime  
✅ **Real-time Updates** - Subscriptions for live data  
✅ **Version-Free** - Add fields without breaking changes  

### How GraphQL Works

```
Client Request → GraphQL Server → Resolvers → Data Sources → Response
```

The client sends a query describing what data it wants, the server processes it and returns exactly that data.

---

## Core Concepts

### 1. Queries - Fetching Data

**What it is:**  
Queries are used to **read data** from a GraphQL server. Think of them like GET requests in REST, but more powerful.

**Why use it:**  
- Fetch exactly the fields you need
- Get related data in one request
- Reduce over-fetching and under-fetching

**Example:**
```graphql
query {
  user(id: "123") {
    id
    name
    email
  }
}
```

**What happens:**  
This retrieves the ID, name, and email of the user with ID `123`. Notice you only request the fields you need - if you don't need `email`, just don't include it!

**Response:**
```json
{
  "data": {
    "user": {
      "id": "123",
      "name": "John Doe",
      "email": "john@example.com"
    }
  }
}
```

---

### 2. Mutations - Modifying Data

**What it is:**  
Mutations are used to **create, update, or delete** data on the server. Think of them like POST/PUT/DELETE in REST.

**Why use it:**  
- Clear distinction between reading and writing
- Can return the modified data in the same request
- Type-safe operations

**Example:**
```graphql
mutation {
  updateOrderStatus(orderId: "456", status: "shipped") {
    orderId
    status
    updatedAt
  }
}
```

**What happens:**  
This updates the status of order `456` to `shipped` and immediately returns the updated order details. You get confirmation of what changed in the same response.

**Response:**
```json
{
  "data": {
    "updateOrderStatus": {
      "orderId": "456",
      "status": "shipped",
      "updatedAt": "2025-01-15T10:30:00Z"
    }
  }
}
```

---

### 3. Subscriptions - Real-time Updates

**What it is:**  
Subscriptions provide **real-time updates** from the server, typically over WebSocket connections. They're like live feeds that push updates to your client.

**Why use it:**  
- No need to poll repeatedly
- Instant updates when data changes
- Perfect for live collaboration, notifications, dashboards

**When to use:**  
- Live chat applications
- Order tracking
- Collaborative editing
- Real-time dashboards
- Notifications

**Example:**
```graphql
subscription {
  orderStatusUpdated(orderId: "456") {
    orderId
    status
    estimatedDelivery
    items {
      name
      quantity
    }
    updatedAt
  }
}
```

**What happens:**  
Once you subscribe, you'll automatically receive updates whenever order `456` changes status. The connection stays open, and updates are pushed to your client instantly.

**Real-world scenario:**  
Imagine tracking a pizza delivery. Instead of refreshing the page every few seconds, the status updates automatically appear on your screen when the pizza moves from "preparing" → "baking" → "out for delivery" → "delivered".

---

### 4. Variables - Making Queries Dynamic

**What it is:**  
Variables let you **parameterize** your queries and mutations, making them reusable with different values.

**Why use it:**  
- Reuse the same query with different values
- Avoid string interpolation (security risk)
- Type-safe parameters

**Example:**
```graphql
query GetUser($userId: ID!, $includeEmail: Boolean = false) {
  user(id: $userId) {
    id
    name
    email @include(if: $includeEmail)
  }
}
```

**Variables (sent separately):**
```json
{
  "userId": "123",
  "includeEmail": true
}
```

**What happens:**  
- `$userId` is **required** (`!` means non-null) - you must provide it
- `$includeEmail` is **optional** (has a default value `false`) - you can omit it
- The query uses these variables instead of hardcoded values

**Why this matters:**  
Instead of writing a new query for each user ID, you write one query and pass different variables. Much cleaner and more secure!

---

### 5. Operation Names - Better Debugging

**What it is:**  
Operation names give your queries and mutations **clear identifiers** for logging and debugging.

**Why use it:**  
- Easier to identify operations in logs
- Better error messages
- Required when using multiple operations

**Example:**
```graphql
query GetUserProfile {
  user(id: "123") {
    id
    name
    email
  }
}

mutation UpdateUserProfile {
  updateUser(id: "123", input: { name: "John Doe" }) {
    id
    name
  }
}
```

**What happens:**  
When you run these operations, logs will show "GetUserProfile" or "UpdateUserProfile" instead of "anonymous query" or "anonymous mutation". Much easier to debug!

---

### 6. Arguments - Filtering and Sorting

**What it is:**  
Arguments let you **pass parameters** to fields for filtering, sorting, pagination, and more.

**Why use it:**  
- Flexible querying
- Server-side filtering (more efficient)
- Standardized pagination

**Example:**
```graphql
query {
  users(
    filter: { role: "admin", active: true }
    sort: { field: "createdAt", order: DESC }
    pagination: { limit: 10, offset: 0 }
  ) {
    id
    name
    email
    createdAt
  }
  
  posts(search: "graphql", tags: ["tutorial", "api"]) {
    id
    title
    author {
      name
    }
  }
}
```

**What happens:**  
- First query: Gets only active admin users, sorted by creation date (newest first), limited to 10 results
- Second query: Searches for posts containing "graphql" with specific tags

**Real-world scenario:**  
Like filtering products on an e-commerce site: "Show me red shirts under $50, sorted by price, first 20 results."

---

## Intermediate Concepts

### 7. Nested Queries - Getting Related Data

**What it is:**  
GraphQL lets you **query nested relationships** in a single request, eliminating multiple round trips.

**Why use it:**  
- Fetch related data in one request
- Reduce network calls
- Better performance

**Example:**
```graphql
query {
  user(id: "123") {
    id
    name
    posts {
      id
      title
      comments {
        id
        text
        author {
          name
        }
      }
    }
    profile {
      bio
      avatar {
        url
        width
        height
      }
    }
  }
}
```

**What happens:**  
In **one request**, you get:
- User information
- All their posts
- Comments on each post
- Authors of those comments
- User's profile with avatar details

**REST comparison:**  
In REST, this would require multiple requests:
1. GET /users/123
2. GET /users/123/posts
3. GET /posts/{id}/comments (for each post)
4. GET /users/123/profile
5. GET /users/123/avatar

GraphQL does it all in **one request**!

---

### 8. Fragments - Reusable Field Sets

**What it is:**  
Fragments are **reusable sets of fields** that you can include in multiple queries to avoid duplication.

**Why use it:**  
- DRY (Don't Repeat Yourself)
- Easier maintenance
- Consistent field selection

**Example:**
```graphql
# Define the fragment once
fragment UserDetails on User {
  id
  name
  email
  createdAt
}

# Use it in multiple places
query GetUserWithPosts {
  user(id: "123") {
    ...UserDetails        # Spread the fragment here
    posts {
      title
    }
  }
}

query GetAllUsers {
  users {
    ...UserDetails        # Reuse the same fragment
  }
}
```

**What happens:**  
Instead of writing `id`, `name`, `email`, `createdAt` in every query, you define it once in a fragment and reuse it. If you need to change which fields are included, you only update the fragment!

**Real-world scenario:**  
Like CSS classes - define the styling once, apply it everywhere. If you need to change the style, update it in one place.

---

### 9. Inline Fragments - Handling Different Types

**What it is:**  
Inline fragments let you **conditionally query fields** based on the actual type of an object. Essential when working with interfaces or unions.

**Why use it:**  
- Handle different types in the same query
- Type-safe field selection
- Works with interfaces and unions

**Example:**
```graphql
query {
  search(query: "graphql") {
    # All results have these common fields (from interface)
    id
    title
    
    # But different types have different fields
    ... on Book {
      author
      isbn
      pages
    }
    ... on Article {
      author {
        name
      }
      publishedAt
      readTime
    }
    ... on Video {
      duration
      thumbnail
    }
  }
}
```

**What happens:**  
The `search` field returns different types (Book, Article, Video). Each type has different fields, so you use inline fragments to request the appropriate fields for each type.

**Real-world scenario:**  
Like a search engine returning different result types: web pages, images, videos. Each type has different metadata you want to display.

---

### 10. Aliases - Querying the Same Field Multiple Times

**What it is:**  
Aliases let you **rename field results**, allowing you to query the same field multiple times with different arguments.

**Why use it:**  
- Query the same field with different parameters
- Avoid conflicts when fetching multiple results
- More flexible queries

**Example:**
```graphql
query {
  userById: user(id: "123") {
    name
  }
  userByEmail: user(email: "john@example.com") {
    name
  }
  recentPosts: posts(limit: 5, sort: "recent") {
    title
  }
  popularPosts: posts(limit: 10, sort: "popularity") {
    title
  }
}
```

**What happens:**  
- `userById` and `userByEmail` are aliases - they rename the `user` field results
- `recentPosts` and `popularPosts` are aliases - they rename the `posts` field results
- You get multiple results from the same field with different arguments

**Response:**
```json
{
  "data": {
    "userById": { "name": "John Doe" },
    "userByEmail": { "name": "John Doe" },
    "recentPosts": [{ "title": "..." }],
    "popularPosts": [{ "title": "..." }]
  }
}
```

**Real-world scenario:**  
Like having multiple filters on a dashboard: "Show me recent orders AND popular products" - both from the same data source but with different criteria.

---

### 11. Directives - Conditional Field Inclusion

**What it is:**  
Directives let you **conditionally include or skip fields**, or modify execution behavior.

**Common directives:**
- `@include(if: Boolean)` - Include field only if condition is true
- `@skip(if: Boolean)` - Skip field if condition is true
- `@deprecated(reason: String)` - Mark field as deprecated

**Why use it:**  
- Dynamic queries based on user permissions
- Feature flags
- API versioning

**Example:**
```graphql
query GetUser(
  $userId: ID!
  $includeEmail: Boolean!
  $skipAddress: Boolean!
) {
  user(id: $userId) {
    id
    name
    email @include(if: $includeEmail)      # Only include if true
    address @skip(if: $skipAddress) {      # Skip if true
      street
      city
    }
    phone @deprecated(reason: "Use contactInfo instead")
    contactInfo {
      phone
      email
    }
  }
}
```

**Variables:**
```json
{
  "userId": "123",
  "includeEmail": true,
  "skipAddress": false
}
```

**What happens:**  
- `email` is included because `$includeEmail` is `true`
- `address` is included because `$skipAddress` is `false` (so we don't skip it)
- `phone` is marked as deprecated (tools will warn you if you use it)

**Real-world scenario:**  
Like feature flags - show premium features only to premium users, or hide sensitive data based on permissions.

---

### 12. Lists and Non-Null Types - Type Safety

**What it is:**  
GraphQL uses `[]` for lists and `!` for non-null types to define **field requirements**.

**Type modifiers:**
- `String` - Nullable string (can be null)
- `String!` - Non-null string (always present)
- `[String]` - Nullable list of nullable strings
- `[String!]` - Nullable list of non-null strings
- `[String!]!` - Non-null list of non-null strings

**Why use it:**  
- Type safety
- Clear contracts
- Better error handling

**Example:**
```graphql
query {
  users {
    id              # ID! - always present
    name            # String! - always present
    email           # String - may be null
    tags            # [String!]! - always an array (never null, may be empty)
    posts           # [Post] - may be null or an array
    friends {      # [User!]! - always an array of users
      id
      name
    }
  }
}
```

**What this means:**
- `id` and `name` are **guaranteed** to be present (non-null)
- `email` **might** be null (user didn't provide it)
- `tags` is **always** an array (never null, may be empty `[]`), and each string inside cannot be null
- `posts` **might** be null OR an array, and each Post inside might also be null
- `friends` is **always** an array (never null, may be empty `[]`), and each User object inside cannot be null

**Breaking down `[String!]!` vs `[User!]!`:**

Both follow the same pattern but with different element types:

| Type | Meaning | Valid Values | Invalid Values |
|------|---------|--------------|----------------|
| `[String!]!` | Non-null list of non-null strings | `[]`, `["tag1", "tag2"]` | `null`, `[null, "tag"]`, `["tag", null]` |
| `[User!]!` | Non-null list of non-null User objects | `[]`, `[{id: "1", name: "John"}]` | `null`, `[null, user]`, `[user, null]` |

**Key points:**
- Both `[String!]!` and `[User!]!` have the **same constraint structure**:
  - Outer `!` = array itself cannot be null
  - Inner `!` = each element cannot be null
- The only difference is the **element type**: `String` vs `User`
- Both can be empty arrays `[]`, but cannot be `null`
- Neither can contain `null` elements

**Detailed examples:**

```graphql
type User {
  tags: [String!]!      # Array of strings
  friends: [User!]!      # Array of User objects
  posts: [Post]          # Array that might be null, elements might be null
  comments: [Comment!]  # Array might be null, but elements cannot be null
}
```

**Valid responses:**
```json
{
  "tags": [],                           // ✅ Empty array
  "tags": ["tag1", "tag2"],            // ✅ Array with strings
  "friends": [],                        // ✅ Empty array
  "friends": [{ "id": "1", "name": "John" }],  // ✅ Array with User objects
  "posts": null,                        // ✅ Allowed (posts is nullable)
  "posts": [null, { "id": "1" }],      // ✅ Allowed (elements can be null)
  "comments": null                      // ✅ Allowed (array can be null)
}
```

**Invalid responses:**
```json
{
  "tags": null,                         // ❌ Array cannot be null
  "tags": [null, "tag"],                // ❌ Elements cannot be null
  "tags": ["tag", null],                // ❌ Elements cannot be null
  "friends": null,                      // ❌ Array cannot be null
  "friends": [null, { "id": "1" }],    // ❌ Elements cannot be null
  "comments": [null, { "id": "1" }]    // ❌ Elements cannot be null (Comment!)
}
```

**Real-world scenario:**  
Like TypeScript types - you know exactly what to expect, and the compiler catches errors before runtime. Think of it like:
- `[String!]!` = A box that always exists (never null), contains strings that always exist (never null)
- `[User!]!` = A box that always exists (never null), contains User objects that always exist (never null)
- The box can be empty `[]`, but it must exist, and everything inside must be valid

---

### 13. Input Types - Complex Arguments

**What it is:**  
Input types are **special object types** used for complex arguments in mutations and queries.

**Why use it:**  
- Pass structured data as arguments
- Type-safe nested data
- Cleaner mutations

**Example:**
```graphql
mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    id
    name
    email
  }
}
```

**Variables:**
```json
{
  "input": {
    "name": "John Doe",
    "email": "john@example.com",
    "password": "secure123",
    "profile": {
      "bio": "Software developer",
      "website": "https://johndoe.com"
    }
  }
}
```

**What happens:**  
Instead of passing many separate arguments, you pass one structured `input` object. Much cleaner, especially for complex operations!

**Real-world scenario:**  
Like filling out a form - instead of passing 20 separate fields, you pass one form object with all the data nested inside.

---

### 14. Enums - Predefined Values

**What it is:**  
Enums define a **set of allowed values** for a field, ensuring type safety.

**Why use it:**  
- Prevent invalid values
- Better IDE autocomplete
- Self-documenting

**Example:**
```graphql
query {
  orders(status: PENDING) {
    id
    status        # Returns: PENDING, PROCESSING, SHIPPED, DELIVERED, CANCELLED
    items {
      product {
        name
        category    # Returns: ELECTRONICS, CLOTHING, BOOKS, FOOD
      }
    }
  }
}

mutation {
  updateOrderStatus(
    orderId: "456"
    status: SHIPPED    # Must be one of the enum values
  ) {
    id
    status
  }
}
```

**What happens:**  
- You can only use valid enum values (`PENDING`, `SHIPPED`, etc.)
- Your IDE will autocomplete the options
- Invalid values are caught before the request is sent

**Real-world scenario:**  
Like a dropdown menu - you can only select from predefined options, preventing typos and invalid data.

---

## Advanced Concepts

### 15. Interfaces - Shared Contracts

**What it is:**  
Interfaces define a **contract** that multiple types can implement, ensuring they share common fields.

**Why use it:**  
- Polymorphism
- Type safety
- Consistent APIs

**Example:**
```graphql
# In the schema
interface SearchResult {
  id: ID!
  title: String!
  description: String!
}

type Book implements SearchResult {
  id: ID!
  title: String!
  description: String!
  author: String!
  isbn: String!
}

type Article implements SearchResult {
  id: ID!
  title: String!
  description: String!
  author: User!
  publishedAt: DateTime!
}
```

**In queries:**
```graphql
query {
  search(query: "graphql") {
    # Common fields from interface
    id
    title
    description
    
    # Type-specific fields
    ... on Book {
      author
      isbn
    }
    ... on Article {
      author {
        name
      }
      publishedAt
    }
  }
}
```

**What happens:**  
All search results (`Book`, `Article`) must have `id`, `title`, and `description` (from the interface), but each type adds its own specific fields.

**Real-world scenario:**  
Like a base class in OOP - all subclasses share common properties, but each adds its own unique features.

---

### 16. Unions - Multiple Possible Types

**What it is:**  
Unions allow a field to return **one of several possible types**, even if those types don't share any common fields or interface. Unlike interfaces (which require shared fields), unions can group completely unrelated types together.

**Key characteristics:**
- A union type represents a value that can be **exactly one** of several object types
- The types in a union don't need to share any common fields
- You **must** use inline fragments (`... on TypeName`) to query union fields
- GraphQL ensures type safety - you can only query fields that exist on the specific type

**Why use it:**  
- **Flexible return types:** When a field can legitimately return different, unrelated types
- **Handle heterogeneous data:** Group different types that serve similar purposes but have different structures
- **Type-safe queries:** GraphQL validates that you only query fields that exist on each type
- **Search results:** Perfect for search APIs that return different content types
- **Error handling:** Can represent success/error states in a type-safe way

**Schema definition:**
```graphql
# Define the individual types
type BlogPost {
  id: ID!
  title: String!
  content: String!
  author: String!
  publishedAt: DateTime!
  tags: [String!]!
}

type Video {
  id: ID!
  title: String!
  url: String!
  duration: Int!  # in seconds
  thumbnail: String!
  views: Int!
}

type Podcast {
  id: ID!
  title: String!
  audioUrl: String!
  episode: Int!
  host: String!
  transcript: String
}

# Define the union type
union Content = BlogPost | Video | Podcast

# Use the union in a field
type Query {
  content(id: ID!): Content
  search(query: String!): [Content!]!
}
```

**In queries - Basic usage:**
```graphql
query {
  content(id: "123") {
    # You MUST use inline fragments - can't query fields directly!
    ... on BlogPost {
      title
      content
      author
      publishedAt
      tags
    }
    ... on Video {
      title
      url
      duration
      thumbnail
      views
    }
    ... on Podcast {
      title
      audioUrl
      episode
      host
      transcript
    }
  }
}
```

**In queries - Handling only specific types:**
```graphql
query {
  search(query: "graphql") {
    # Only handle the types you care about
    ... on BlogPost {
      title
      author
      publishedAt
    }
    ... on Video {
      title
      duration
      thumbnail
    }
    # Podcast results will be ignored if not handled
  }
}
```

**In queries - Using named fragments (cleaner for complex queries):**
```graphql
query {
  search(query: "tutorial") {
    ...BlogPostFields
    ...VideoFields
    ...PodcastFields
  }
}

fragment BlogPostFields on BlogPost {
  id
  title
  author
  publishedAt
}

fragment VideoFields on Video {
  id
  title
  url
  duration
  thumbnail
}

fragment PodcastFields on Podcast {
  id
  title
  episode
  host
}
```

**Common patterns:**

**1. Success/Error pattern:**
```graphql
union Result = Success | Error

type Success {
  data: String!
  message: String
}

type Error {
  code: String!
  message: String!
  details: String
}

type Query {
  operation: Result
}

# Query example
query {
  operation {
    ... on Success {
      data
      message
    }
    ... on Error {
      code
      message
      details
    }
  }
}
```

**2. Search results pattern:**
```graphql
union SearchResult = User | Post | Comment | Tag

type Query {
  search(query: String!): [SearchResult!]!
}
```

**3. Event/Notification pattern:**
```graphql
union Notification = EmailNotification | PushNotification | SMSNotification

type EmailNotification {
  to: String!
  subject: String!
  body: String!
}

type PushNotification {
  deviceId: String!
  title: String!
  message: String!
}

type SMSNotification {
  phoneNumber: String!
  message: String!
}
```

**What happens:**  
When you query a union field:
1. GraphQL returns **exactly one** of the union member types
2. You **must** use inline fragments (`... on TypeName`) to access fields
3. GraphQL validates that you only query fields that exist on that specific type
4. If you don't handle a particular type, those results are simply ignored (no error)

**Important rules:**
- ❌ **Can't query fields directly:** `content { title }` - This will error!
- ✅ **Must use fragments:** `content { ... on BlogPost { title } }`
- ✅ **Can query common fields:** If types happen to share field names, you still need fragments
- ✅ **Type safety:** GraphQL ensures you only query valid fields for each type

**Interfaces vs Unions - When to use which:**

| Feature | Interfaces | Unions |
|---------|-----------|--------|
| **Common fields** | ✅ Required - all types share fields | ❌ Not required - types can be completely different |
| **Query syntax** | Can query common fields directly | Must use fragments for all fields |
| **Use case** | Types with shared structure | Types grouped by purpose, not structure |
| **Example** | `SearchResult` with `id`, `title`, `description` | `Content` that can be `BlogPost`, `Video`, or `Podcast` |

**When to use Interfaces:**
- Types share common fields (e.g., all have `id`, `title`, `createdAt`)
- You want to query common fields without fragments
- Types represent variations of the same concept

**When to use Unions:**
- Types are completely different but serve similar purposes
- Types don't share common fields
- You need maximum flexibility in return types
- Representing success/error states or different event types

**Real-world scenarios:**

1. **Media library:** A content management system where `content` can be a blog post, video, podcast, or image gallery - all completely different structures
2. **Search results:** A search API returning users, posts, comments, and tags - unrelated types grouped by search context
3. **Event system:** Notifications that can be emails, push notifications, or SMS - different structures, same purpose
4. **API responses:** A field that can return either success data or error information
5. **File system:** A file that can be a document, image, video, or folder - different metadata for each type

**Best practices:**
- ✅ Use unions when types are conceptually related but structurally different
- ✅ Always handle all union member types in your queries (or explicitly ignore some)
- ✅ Use named fragments for complex union queries to keep code clean
- ✅ Consider using `__typename` to identify the actual type returned
- ❌ Don't use unions just to avoid creating an interface - if types share fields, use interfaces
- ❌ Don't create unions with too many member types (5+ becomes hard to manage)

**Using `__typename` to identify types:**
```graphql
query {
  content(id: "123") {
    __typename  # Returns "BlogPost", "Video", or "Podcast"
    ... on BlogPost {
      title
      author
    }
    ... on Video {
      title
      duration
    }
    ... on Podcast {
      title
      host
    }
  }
}
```

**Real-world scenario:**  
Like a media player that can play different file types (MP3, MP4, WAV) - they're completely different formats with different properties (bitrate, codec, sample rate), but you handle them all in one player interface. Each file type needs different handling, but they all serve the same purpose: playing media.

---

### 17. Scalar Types - Primitive Values

**What it is:**  
Scalar types represent **primitive values**. GraphQL provides built-in scalars and allows custom ones.

**Built-in scalars:**
- `Int` - 32-bit integer
- `Float` - Double-precision floating-point
- `String` - UTF-8 character sequence
- `Boolean` - true or false
- `ID` - Unique identifier (serialized as String)

**Custom scalars:**
- `DateTime` - ISO 8601 date string
- `JSON` - Arbitrary JSON object
- `URL` - Valid URL string
- `Email` - Valid email address

**Example:**
```graphql
query {
  user(id: "123") {
    id              # ID scalar: unique identifier
    name            # String scalar: text
    age             # Int scalar: whole number
    salary          # Float scalar: decimal number
    isActive        # Boolean scalar: true/false
    createdAt       # DateTime scalar (custom): ISO 8601 date
    metadata        # JSON scalar (custom): arbitrary JSON object
  }
}
```

**What happens:**  
Each field has a specific type that determines what values are valid and how they're serialized.

---

### 18. Error Handling - Graceful Failures

**What it is:**  
GraphQL returns **errors alongside data**, allowing partial results even when some fields fail.

**Why it's powerful:**
- Partial data is still useful
- Detailed error information
- Better user experience

**Example Query:**
```graphql
query {
  user(id: "123") {
    id
    name
    email
    posts {
      id
      title
    }
  }
}
```

**Example Response (with errors):**
```json
{
  "data": {
    "user": {
      "id": "123",
      "name": "John Doe",
      "email": null,        # Field failed
      "posts": null         # Field failed
    }
  },
  "errors": [
    {
      "message": "Email is private",
      "path": ["user", "email"],
      "extensions": {
        "code": "FORBIDDEN",
        "field": "email"
      }
    },
    {
      "message": "Failed to fetch posts",
      "path": ["user", "posts"],
      "extensions": {
        "code": "INTERNAL_ERROR"
      }
    }
  ]
}
```

**What happens:**  
- You still get the user's `id` and `name` (partial success)
- `email` and `posts` failed, but you know exactly why
- Each error includes the `path` showing where it occurred

**Real-world scenario:**  
Like a dashboard that shows what it can - if one widget fails, the others still work. Much better than the whole page breaking!

---

### 19. Introspection - Schema Discovery

**What it is:**  
Introspection lets you **query the GraphQL schema itself** to discover available types, fields, and operations.

**Why use it:**
- Build tools and IDEs
- Generate documentation
- Validate queries

**Example:**
```graphql
query IntrospectSchema {
  __schema {
    types {
      name
      kind
      fields {
        name
        type {
          name
          kind
        }
      }
    }
    queryType {
      name
      fields {
        name
        description
      }
    }
  }
  
  __type(name: "User") {
    name
    fields {
      name
      type {
        name
        kind
      }
    }
  }
}
```

**What happens:**  
You can programmatically discover:
- What types exist
- What fields each type has
- What queries and mutations are available
- Field types and descriptions

**Real-world scenario:**  
This is how GraphQL IDEs (like GraphiQL) work - they introspect the schema to provide autocomplete and documentation.

---

### 20. Multiple Operations - One Document, Many Options

**What it is:**  
You can define **multiple operations** in a single document, but only execute one per request.

**Why use it:**
- Organize related operations
- Share fragments
- Better code organization

**Example:**
```graphql
query GetUser {
  user(id: "123") {
    id
    name
  }
}

query GetPosts {
  posts {
    id
    title
  }
}

mutation CreatePost {
  createPost(input: { title: "New Post" }) {
    id
    title
  }
}
```

**What happens:**  
You define all three operations in one file, but when you make a request, you specify which one to execute using the operation name.

**Real-world scenario:**  
Like having multiple functions in one file - you define them all together, but call them individually when needed.

---

### 21. Field Selection - Request Only What You Need

**What it is:**  
GraphQL lets you **request exactly the fields you need**, preventing over-fetching and under-fetching.

**Why it matters:**
- Smaller payloads
- Faster responses
- Better performance

**Example:**
```graphql
# Minimal query - mobile app, just need basic info
query GetUserBasic {
  user(id: "123") {
    id
    name
  }
}

# Detailed query - admin dashboard, need everything
query GetUserDetailed {
  user(id: "123") {
    id
    name
    email
    bio
    avatar {
      url
      width
      height
    }
    posts {
      id
      title
      createdAt
      comments {
        id
        text
        author {
          name
        }
      }
    }
    followers {
      id
      name
    }
    following {
      id
      name
    }
  }
}
```

**What happens:**  
- Mobile app uses `GetUserBasic` - small payload, fast loading
- Admin dashboard uses `GetUserDetailed` - comprehensive data

**REST comparison:**  
In REST, you'd get the same large response regardless of what you need. GraphQL lets you tailor the response to your use case.

---

### 22. Resolvers - How GraphQL Executes Queries

**What it is:**  
Resolvers are **functions that resolve field values** in your GraphQL schema. They're the bridge between your GraphQL schema and your data sources (databases, APIs, etc.).

**Why it matters:**
- Understand how GraphQL actually works under the hood
- Customize data fetching logic
- Implement business logic
- Connect to any data source

**How it works:**
```javascript
// Schema
type Query {
  user(id: ID!): User
  posts: [Post!]!
}

type User {
  id: ID!
  name: String!
  email: String!
  posts: [Post!]!  # Nested field - needs its own resolver
}

// Resolvers
const resolvers = {
  Query: {
    // Resolver for Query.user
    user: async (parent, args, context) => {
      // parent: result from parent resolver (null for root queries)
      // args: { id: "123" }
      // context: shared data (auth, db connection, etc.)
      return await context.db.users.findById(args.id);
    },
    
    // Resolver for Query.posts
    posts: async (parent, args, context) => {
      return await context.db.posts.findAll();
    }
  },
  
  User: {
    // Resolver for User.posts (nested field)
    posts: async (parent, args, context) => {
      // parent: the User object from parent resolver
      return await context.db.posts.findByUserId(parent.id);
    }
  }
};
```

**Resolver function signature:**
```javascript
(parent, args, context, info) => {
  // parent: Result from parent resolver
  // args: Arguments passed to the field
  // context: Shared context (auth, db, etc.)
  // info: Query metadata (field name, AST, etc.)
  return value; // Return the field value
}
```

**Example - Complex resolver:**
```javascript
const resolvers = {
  Query: {
    user: async (parent, args, context) => {
      // Check authentication
      if (!context.user) {
        throw new Error('Unauthorized');
      }
      
      // Fetch user from database
      const user = await context.db.users.findById(args.id);
      
      // Check permissions
      if (user.id !== context.user.id && !context.user.isAdmin) {
        throw new Error('Forbidden');
      }
      
      return user;
    }
  },
  
  User: {
    // Computed field - not in database
    fullName: (parent) => {
      return `${parent.firstName} ${parent.lastName}`;
    },
    
    // Field with custom logic
    email: async (parent, args, context) => {
      // Only show email if user is viewing their own profile
      if (context.user?.id === parent.id) {
        return parent.email;
      }
      return null; // Hide email from others
    },
    
    // Async field resolution
    posts: async (parent, args, context) => {
      return await context.db.posts.findByUserId(parent.id);
    }
  }
};
```

**Real-world scenario:**  
Like a waiter in a restaurant - the query is the order, resolvers are the waiters who fetch each dish (field) from different kitchens (data sources) and bring them together.

---

### 23. Custom Scalars - Extending GraphQL Types

**What it is:**  
Custom scalars let you define **new primitive types** beyond the built-in ones (Int, String, Boolean, etc.). Useful for dates, URLs, emails, JSON, and other specialized types.

**Why use it:**
- Type safety for specialized data
- Validation at the schema level
- Better serialization/deserialization
- Self-documenting APIs

**Schema definition:**
```graphql
# Define custom scalar
scalar DateTime
scalar Email
scalar URL
scalar JSON

type User {
  id: ID!
  email: Email!           # Custom scalar - validates email format
  website: URL            # Custom scalar - validates URL format
  createdAt: DateTime!    # Custom scalar - ISO 8601 date
  metadata: JSON          # Custom scalar - arbitrary JSON object
}
```

**Implementation (JavaScript/Node.js example):**
```javascript
const { GraphQLScalarType } = require('graphql');

// DateTime scalar
const DateTime = new GraphQLScalarType({
  name: 'DateTime',
  description: 'ISO 8601 date-time string',
  
  // Serialize: convert value to send to client
  serialize(value) {
    if (value instanceof Date) {
      return value.toISOString();
    }
    return value;
  },
  
  // ParseValue: convert from variable
  parseValue(value) {
    return new Date(value);
  },
  
  // ParseLiteral: convert from query AST
  parseLiteral(ast) {
    if (ast.kind === Kind.STRING) {
      return new Date(ast.value);
    }
    return null;
  }
});

// Email scalar with validation
const Email = new GraphQLScalarType({
  name: 'Email',
  description: 'Valid email address',
  
  serialize(value) {
    return value;
  },
  
  parseValue(value) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(value)) {
      throw new Error('Invalid email format');
    }
    return value;
  },
  
  parseLiteral(ast) {
    if (ast.kind === Kind.STRING) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(ast.value)) {
        throw new Error('Invalid email format');
      }
      return ast.value;
    }
    return null;
  }
});

// JSON scalar for arbitrary JSON objects
const JSON = new GraphQLScalarType({
  name: 'JSON',
  description: 'Arbitrary JSON object',
  
  serialize(value) {
    return value;
  },
  
  parseValue(value) {
    return value;
  },
  
  parseLiteral(ast) {
    // Parse JSON from string literal
    if (ast.kind === Kind.STRING) {
      return JSON.parse(ast.value);
    }
    return null;
  }
});

// Export scalars
module.exports = {
  DateTime,
  Email,
  JSON
};
```

**Usage in queries:**
```graphql
query {
  user(id: "123") {
    email          # Email scalar - validated
    website        # URL scalar - validated
    createdAt      # DateTime scalar - ISO 8601 format
    metadata       # JSON scalar - arbitrary object
  }
}

mutation {
  createUser(
    email: "john@example.com"  # Email validation happens here
    metadata: "{\"key\": \"value\"}"  # JSON string parsed
  ) {
    id
    email
  }
}
```

**Common custom scalars:**
- `DateTime` - ISO 8601 dates
- `Date` - Date only (no time)
- `Email` - Validated email addresses
- `URL` - Validated URLs
- `JSON` - Arbitrary JSON objects
- `UUID` - UUID strings
- `BigInt` - Large integers
- `Decimal` - Precise decimal numbers

**Real-world scenario:**  
Like having specialized containers for different types of items - a DateTime container ensures dates are always in the right format, an Email container validates email addresses automatically.

---

### 24. Default Values - Optional Arguments with Fallbacks

**What it is:**  
Default values let you provide **fallback values** for optional arguments, making queries simpler and more flexible.

**Why use it:**
- Simpler queries (don't need to specify every argument)
- Sensible defaults for common cases
- Backward compatibility
- Better developer experience

**Schema definition:**
```graphql
type Query {
  # Default values for optional arguments
  posts(
    limit: Int = 10           # Default: 10 posts
    offset: Int = 0            # Default: start from beginning
    sortBy: String = "date"   # Default: sort by date
    order: SortOrder = DESC    # Default: descending order
  ): [Post!]!
  
  users(
    role: UserRole = USER      # Default: regular users
    active: Boolean = true     # Default: only active users
  ): [User!]!
}

enum SortOrder {
  ASC
  DESC
}

enum UserRole {
  USER
  ADMIN
  MODERATOR
}
```

**Usage examples:**
```graphql
# Use all defaults
query {
  posts {  # Gets 10 posts, offset 0, sorted by date DESC
    id
    title
  }
}

# Override some defaults
query {
  posts(limit: 20, sortBy: "popularity") {  # 20 posts, sorted by popularity
    id
    title
  }
}

# Override all defaults
query {
  posts(limit: 5, offset: 10, sortBy: "title", order: ASC) {
    id
    title
  }
}
```

**Default values with variables:**
```graphql
query GetPosts(
  $limit: Int = 10        # Default in query definition
  $offset: Int = 0
) {
  posts(limit: $limit, offset: $offset) {
    id
    title
  }
}

# Variables (can omit defaults)
{
  "limit": 20  # offset defaults to 0
}

# Or omit entirely - both use defaults
{}
```

**Complex default values:**
```graphql
input PostFilter {
  tags: [String!] = []           # Default: empty array
  published: Boolean = true       # Default: only published
  minViews: Int = 0              # Default: no minimum
}

type Query {
  posts(filter: PostFilter = {}) {  # Default: empty filter object
    id
    title
  }
}
```

**Real-world scenario:**  
Like a vending machine with default settings - you can press a button for the default snack, or customize your selection. Defaults make common cases easy while still allowing full control.

---

### 25. Cursor-Based Pagination - Efficient Large Datasets

**What it is:**  
Cursor-based pagination uses **opaque cursors** (pointers to specific records) instead of offset/limit, making it more efficient and reliable for large datasets.

**Why use it:**
- More efficient (no need to skip records)
- Consistent results (no duplicates when data changes)
- Works better with real-time data
- Industry standard (used by GitHub, Twitter, etc.)

**Offset vs Cursor comparison:**

| Feature | Offset-based | Cursor-based |
|---------|-------------|--------------|
| **Efficiency** | Slow for large offsets | Fast (direct lookup) |
| **Consistency** | Can skip/duplicate items | Always consistent |
| **Real-time safe** | No (data changes affect results) | Yes (stable results) |
| **Complexity** | Simple | More complex |

**Schema definition:**
```graphql
# Connection pattern (Relay-style)
type Query {
  posts(first: Int, after: String): PostConnection!
}

type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
}

type PostEdge {
  node: Post!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

type Post {
  id: ID!
  title: String!
  content: String!
  createdAt: DateTime!
}
```

**Query example:**
```graphql
query {
  posts(first: 10, after: "cursor123") {
    edges {
      node {
        id
        title
        createdAt
      }
      cursor
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      startCursor
      endCursor
    }
  }
}
```

**Response:**
```json
{
  "data": {
    "posts": {
      "edges": [
        {
          "node": {
            "id": "1",
            "title": "Post 1",
            "createdAt": "2025-01-01T00:00:00Z"
          },
          "cursor": "eyJpZCI6MX0="
        },
        {
          "node": {
            "id": "2",
            "title": "Post 2",
            "createdAt": "2025-01-02T00:00:00Z"
          },
          "cursor": "eyJpZCI6Mn0="
        }
      ],
      "pageInfo": {
        "hasNextPage": true,
        "hasPreviousPage": false,
        "startCursor": "eyJpZCI6MX0=",
        "endCursor": "eyJpZCI6Mn0="
      }
    }
  }
}
```

**Simpler cursor pattern (non-Relay):**
```graphql
type Query {
  posts(first: Int, after: String): PostsPage!
}

type PostsPage {
  items: [Post!]!
  nextCursor: String
  hasMore: Boolean!
}
```

**Implementation example:**
```javascript
// Resolver implementation
const resolvers = {
  Query: {
    posts: async (parent, args, context) => {
      const { first = 10, after } = args;
      
      // Decode cursor (base64 encoded JSON)
      let cursor = null;
      if (after) {
        cursor = JSON.parse(Buffer.from(after, 'base64').toString());
      }
      
      // Fetch one extra to check if there's more
      const limit = first + 1;
      const posts = await context.db.posts.findAfter(cursor, limit);
      
      const hasMore = posts.length > first;
      const items = posts.slice(0, first);
      
      return {
        items,
        nextCursor: hasMore ? encodeCursor(items[items.length - 1]) : null,
        hasMore
      };
    }
  }
};

// Helper functions
function encodeCursor(post) {
  return Buffer.from(JSON.stringify({ id: post.id, createdAt: post.createdAt }))
    .toString('base64');
}

function decodeCursor(cursor) {
  return JSON.parse(Buffer.from(cursor, 'base64').toString());
}
```

**Bidirectional pagination:**
```graphql
type Query {
  posts(
    first: Int
    after: String
    last: Int
    before: String
  ): PostConnection!
}
```

**Real-world scenario:**  
Like reading a book with bookmarks - instead of saying "skip 100 pages" (offset), you use a bookmark (cursor) that points to exactly where you left off. Even if pages are added or removed, your bookmark still works.

---

### 26. Custom Directives - Extending GraphQL Behavior

**What it is:**  
Custom directives let you **extend GraphQL's execution behavior** with custom logic. While built-in directives (`@include`, `@skip`) modify query execution, custom directives can add authentication, caching, rate limiting, and more.

**Why use it:**
- Reusable cross-cutting concerns
- Declarative behavior in schema
- Clean separation of concerns
- Powerful extension mechanism

**Schema definition:**
```graphql
# Define custom directive
directive @auth(requires: Role = USER) on FIELD_DEFINITION
directive @cache(maxAge: Int) on FIELD_DEFINITION
directive @rateLimit(max: Int, window: String) on FIELD_DEFINITION
directive @deprecated(reason: String) on FIELD_DEFINITION | ENUM_VALUE

enum Role {
  USER
  ADMIN
  MODERATOR
}

type Query {
  user(id: ID!): User @auth(requires: USER)
  adminData: String @auth(requires: ADMIN)
  posts: [Post!]! @cache(maxAge: 3600) @rateLimit(max: 100, window: "1h")
}

type User {
  id: ID!
  email: String @auth(requires: ADMIN)  # Only admins can see emails
  name: String!
}
```

**Implementation (Apollo Server example):**
```javascript
const { mapSchema, getDirective, MapperKind } = require('@graphql-tools/utils');
const { defaultFieldResolver } = require('graphql');

// Auth directive
function authDirectiveTransformer(schema) {
  return mapSchema(schema, {
    [MapperKind.OBJECT_FIELD]: (fieldConfig) => {
      const authDirective = getDirective(schema, fieldConfig, 'auth')?.[0];
      
      if (authDirective) {
        const { requires } = authDirective;
        const { resolve = defaultFieldResolver } = fieldConfig;
        
        fieldConfig.resolve = async (parent, args, context, info) => {
          // Check authentication
          if (!context.user) {
            throw new Error('Unauthorized');
          }
          
          // Check role
          if (!hasRole(context.user, requires)) {
            throw new Error('Forbidden');
          }
          
          return resolve(parent, args, context, info);
        };
      }
      
      return fieldConfig;
    }
  });
}

// Cache directive
function cacheDirectiveTransformer(schema) {
  return mapSchema(schema, {
    [MapperKind.OBJECT_FIELD]: (fieldConfig) => {
      const cacheDirective = getDirective(schema, fieldConfig, 'cache')?.[0];
      
      if (cacheDirective) {
        const { maxAge } = cacheDirective;
        const { resolve = defaultFieldResolver } = fieldConfig;
        const cache = new Map();
        
        fieldConfig.resolve = async (parent, args, context, info) => {
          const cacheKey = JSON.stringify({ parent, args, info });
          const cached = cache.get(cacheKey);
          
          if (cached && Date.now() - cached.timestamp < maxAge * 1000) {
            return cached.value;
          }
          
          const result = await resolve(parent, args, context, info);
          cache.set(cacheKey, { value: result, timestamp: Date.now() });
          
          return result;
        };
      }
      
      return fieldConfig;
    }
  });
}

// Rate limit directive
function rateLimitDirectiveTransformer(schema) {
  return mapSchema(schema, {
    [MapperKind.OBJECT_FIELD]: (fieldConfig) => {
      const rateLimitDirective = getDirective(schema, fieldConfig, 'rateLimit')?.[0];
      
      if (rateLimitDirective) {
        const { max, window } = rateLimitDirective;
        const { resolve = defaultFieldResolver } = fieldConfig;
        const requestCounts = new Map();
        
        fieldConfig.resolve = async (parent, args, context, info) => {
          const key = context.user?.id || context.ip;
          const now = Date.now();
          const windowMs = parseWindow(window);
          
          // Clean old entries
          const userRequests = requestCounts.get(key) || [];
          const recentRequests = userRequests.filter(
            time => now - time < windowMs
          );
          
          if (recentRequests.length >= max) {
            throw new Error('Rate limit exceeded');
          }
          
          recentRequests.push(now);
          requestCounts.set(key, recentRequests);
          
          return resolve(parent, args, context, info);
        };
      }
      
      return fieldConfig;
    }
  });
}
```

**Usage in queries:**
```graphql
# Directives work automatically - no special syntax needed
query {
  user(id: "123") {  # @auth directive checks permissions
    id
    name
    email  # @auth(requires: ADMIN) - only admins see this
  }
  
  posts {  # @cache and @rateLimit directives applied
    id
    title
  }
}
```

**Directive locations:**
- `FIELD_DEFINITION` - On field definitions
- `OBJECT` - On object types
- `ARGUMENT_DEFINITION` - On arguments
- `QUERY` - On query operations
- `MUTATION` - On mutation operations
- `SUBSCRIPTION` - On subscription operations
- `ENUM_VALUE` - On enum values

**Real-world scenario:**  
Like decorators in programming - you add `@auth` to secure a field, `@cache` to speed it up, `@rateLimit` to protect it, all declaratively in your schema without cluttering your resolver code.

---

### 27. Context - Sharing Data Across Resolvers

**What it is:**  
Context is a **shared object** passed to every resolver, containing data like authentication info, database connections, and other shared resources.

**Why use it:**
- Share data across resolvers
- Avoid global variables
- Dependency injection
- Testability

**Setting up context:**
```javascript
// Apollo Server example
const server = new ApolloServer({
  typeDefs,
  resolvers,
  context: ({ req }) => {
    // Extract auth token from request
    const token = req.headers.authorization?.replace('Bearer ', '');
    const user = token ? verifyToken(token) : null;
    
    return {
      user,                    // Current authenticated user
      db: database,            // Database connection
      cache: redisClient,      // Cache client
      logger: logger,         // Logger instance
      ip: req.ip,             // Client IP
      requestId: req.id       // Request ID for tracing
    };
  }
});
```

**Using context in resolvers:**
```javascript
const resolvers = {
  Query: {
    user: async (parent, args, context) => {
      // Access user from context
      if (!context.user) {
        throw new Error('Unauthorized');
      }
      
      // Use database from context
      return await context.db.users.findById(args.id);
    },
    
    posts: async (parent, args, context) => {
      // Check permissions
      const isAdmin = context.user?.role === 'ADMIN';
      
      // Use cache from context
      const cacheKey = `posts:${args.limit}`;
      const cached = await context.cache.get(cacheKey);
      
      if (cached) {
        return cached;
      }
      
      // Fetch from database
      const posts = await context.db.posts.findAll({
        limit: args.limit,
        includePrivate: isAdmin
      });
      
      // Cache result
      await context.cache.set(cacheKey, posts, 3600);
      
      // Log with logger from context
      context.logger.info('Fetched posts', { count: posts.length });
      
      return posts;
    }
  },
  
  User: {
    email: async (parent, args, context) => {
      // Only show email to owner or admins
      if (context.user?.id === parent.id || context.user?.role === 'ADMIN') {
        return parent.email;
      }
      return null;
    }
  }
};
```

**Context with subscriptions:**
```javascript
const server = new ApolloServer({
  typeDefs,
  resolvers,
  context: async ({ req, connection }) => {
    // HTTP requests (queries/mutations)
    if (req) {
      const token = req.headers.authorization?.replace('Bearer ', '');
      return {
        user: token ? verifyToken(token) : null,
        db: database
      };
    }
    
    // WebSocket connections (subscriptions)
    if (connection) {
      return {
        user: connection.context.user,
        db: database
      };
    }
  },
  subscriptions: {
    onConnect: async (connectionParams) => {
      const token = connectionParams.authorization;
      const user = token ? verifyToken(token) : null;
      
      return { user };
    }
  }
});
```

**Best practices:**
- ✅ Put shared resources in context (db, cache, logger)
- ✅ Include authentication info (user, permissions)
- ✅ Keep context immutable (don't mutate it)
- ✅ Use context for dependency injection
- ❌ Don't put request-specific data that changes
- ❌ Don't put large objects that aren't needed

**Real-world scenario:**  
Like a toolbox that every worker (resolver) has access to - they all share the same tools (database, cache, logger) without needing to carry their own copies.

---

### 28. Query Complexity Analysis - Protecting Your API

**What it is:**  
Query complexity analysis **calculates the cost** of a query before execution, preventing expensive queries that could overload your server.

**Why use it:**
- Prevent DoS attacks
- Protect against expensive queries
- Set query limits
- Better resource management

**How it works:**
```javascript
// Define complexity costs
const complexityLimit = 1000;

const complexityDirective = {
  // Simple field = 1 point
  // List field = N * (item complexity)
  // Nested field = parent complexity * child complexity
};

// Example complexity calculation
query {
  users {           # 10 users * complexity
    posts {         # 5 posts each * complexity
      comments {    # 10 comments each * complexity
        author {    # 1 author each
          name
        }
      }
    }
  }
}
// Total: 10 * 5 * 10 * 1 = 500 complexity points
```

**Implementation:**
```javascript
const { createComplexityLimitRule } = require('graphql-query-complexity');

const complexityRule = createComplexityLimitRule({
  maximumComplexity: 1000,
  variables: {},
  onComplete: (complexity) => {
    console.log(`Query complexity: ${complexity}`);
  },
  estimators: [
    // Simple field = 1
    (options) => {
      return options.field.complexity || 1;
    },
    // List field = multiplier * item complexity
    (options) => {
      if (options.field.type.toString().includes('[')) {
        return options.childComplexity * (options.args.first || options.args.limit || 10);
      }
    }
  ]
});

// Apply to Apollo Server
const server = new ApolloServer({
  typeDefs,
  resolvers,
  validationRules: [complexityRule]
});
```

**Schema-level complexity:**
```graphql
type Query {
  users: [User!]!  # Complexity: 10 (assumes 10 users)
  user(id: ID!): User  # Complexity: 1
  posts: [Post!]!  # Complexity: 20 (assumes 20 posts)
}

type User {
  id: ID!
  name: String!  # Complexity: 1
  posts: [Post!]!  # Complexity: 5 (assumes 5 posts per user)
  friends: [User!]!  # Complexity: 10 (assumes 10 friends)
}
```

**Custom complexity:**
```javascript
const resolvers = {
  Query: {
    users: {
      complexity: ({ args, childComplexity }) => {
        const limit = args.limit || 10;
        return limit * childComplexity;
      },
      resolve: async (parent, args, context) => {
        return await context.db.users.findAll({ limit: args.limit });
      }
    }
  },
  
  User: {
    posts: {
      complexity: ({ args, childComplexity }) => {
        const limit = args.limit || 5;
        return limit * childComplexity;
      },
      resolve: async (parent, args, context) => {
        return await context.db.posts.findByUserId(parent.id, { limit: args.limit });
      }
    }
  }
};
```

**Error response:**
```json
{
  "errors": [
    {
      "message": "Query complexity of 1500 exceeds maximum of 1000",
      "extensions": {
        "code": "QUERY_COMPLEXITY_EXCEEDED",
        "complexity": 1500,
        "maximum": 1000
      }
    }
  ]
}
```

**Best practices:**
- ✅ Set reasonable complexity limits (100-1000)
- ✅ Consider list sizes in calculations
- ✅ Monitor complexity in production
- ✅ Provide helpful error messages
- ✅ Adjust limits based on server capacity

**Real-world scenario:**  
Like a bouncer at a club checking IDs - complexity analysis checks if a query is "too expensive" before letting it in, protecting your server from being overwhelmed.

---

### 29. File Uploads - Handling Binary Data

**What it is:**  
GraphQL can handle **file uploads** using the multipart request specification or by encoding files as base64 strings.

**Why it matters:**
- Upload images, documents, videos
- Profile pictures, avatars
- Document attachments
- Media content

**Method 1: Multipart Request (Recommended)**
```graphql
# Schema
scalar Upload

type Mutation {
  uploadAvatar(file: Upload!): String!
  uploadDocument(file: Upload!, description: String): Document!
}

type Document {
  id: ID!
  filename: String!
  url: String!
  size: Int!
  mimeType: String!
}
```

**Client-side (using apollo-upload-client):**
```javascript
import { gql } from '@apollo/client';
import { createUploadLink } from 'apollo-upload-client';

const UPLOAD_AVATAR = gql`
  mutation UploadAvatar($file: Upload!) {
    uploadAvatar(file: $file)
  }
`;

// Usage
const fileInput = document.querySelector('input[type="file"]');
const file = fileInput.files[0];

const { data } = await client.mutate({
  mutation: UPLOAD_AVATAR,
  variables: { file }
});
```

**Server-side (Apollo Server):**
```javascript
const { ApolloServer } = require('apollo-server-express');
const { GraphQLUpload } = require('graphql-upload');
const fs = require('fs').promises;
const path = require('path');

const resolvers = {
  Upload: GraphQLUpload,
  
  Mutation: {
    uploadAvatar: async (parent, { file }, context) => {
      const { createReadStream, filename, mimetype } = await file;
      
      // Validate file type
      if (!mimetype.startsWith('image/')) {
        throw new Error('File must be an image');
      }
      
      // Validate file size (e.g., 5MB max)
      const maxSize = 5 * 1024 * 1024; // 5MB
      // Note: You'd need to check size from stream or separately
      
      // Generate unique filename
      const uniqueFilename = `${Date.now()}-${filename}`;
      const filepath = path.join(__dirname, 'uploads', uniqueFilename);
      
      // Save file
      const stream = createReadStream();
      await new Promise((resolve, reject) => {
        stream
          .pipe(fs.createWriteStream(filepath))
          .on('finish', resolve)
          .on('error', reject);
      });
      
      // Return URL
      return `/uploads/${uniqueFilename}`;
    },
    
    uploadDocument: async (parent, { file, description }, context) => {
      const { createReadStream, filename, mimetype } = await file;
      
      // Save to cloud storage (e.g., S3, Cloudinary)
      const url = await uploadToS3(createReadStream(), filename);
      
      // Save metadata to database
      const document = await context.db.documents.create({
        filename,
        url,
        mimeType: mimetype,
        description,
        userId: context.user.id
      });
      
      return document;
    }
  }
};
```

**Method 2: Base64 Encoding**
```graphql
type Mutation {
  uploadAvatar(base64File: String!, filename: String!): String!
}
```

```javascript
const resolvers = {
  Mutation: {
    uploadAvatar: async (parent, { base64File, filename }, context) => {
      // Decode base64
      const buffer = Buffer.from(base64File, 'base64');
      
      // Save file
      const filepath = path.join(__dirname, 'uploads', filename);
      await fs.writeFile(filepath, buffer);
      
      return `/uploads/${filename}`;
    }
  }
};
```

**File validation:**
```javascript
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/gif'];
const MAX_SIZE = 5 * 1024 * 1024; // 5MB

async function validateFile(file) {
  const { mimetype, encoding } = await file;
  
  // Check MIME type
  if (!ALLOWED_TYPES.includes(mimetype)) {
    throw new Error(`File type ${mimetype} not allowed`);
  }
  
  // Check size (if available)
  // Note: Size checking depends on your implementation
  
  return true;
}
```

**Real-world scenario:**  
Like a mailbox that accepts packages - GraphQL can accept file uploads just like it accepts data, allowing you to upload profile pictures, documents, or any binary content through your API.

---

### 30. DataLoader - Batching and Caching for Performance

**What it is:**  
DataLoader is a **batching and caching utility** that solves the N+1 query problem by batching multiple requests and caching results within a single request.

**The N+1 Problem:**
```graphql
query {
  posts {
    id
    title
    author {      # N queries - one per post!
      id
      name
    }
  }
}
# If you have 100 posts, this makes 101 queries:
# 1 query for posts + 100 queries for authors
```

**Why use DataLoader:**
- Eliminates N+1 queries
- Batches requests automatically
- Caches results per request
- Dramatically improves performance

**How it works:**
```javascript
const DataLoader = require('dataloader');

// Create a DataLoader for users
const userLoader = new DataLoader(async (userIds) => {
  // This function receives an array of IDs
  // and returns an array of users in the same order
  const users = await db.users.findByIds(userIds);
  
  // Create a map for quick lookup
  const userMap = new Map();
  users.forEach(user => {
    userMap.set(user.id, user);
  });
  
  // Return users in the same order as requested IDs
  return userIds.map(id => userMap.get(id) || null);
});

// Usage in resolver
const resolvers = {
  Post: {
    author: async (parent, args, context) => {
      // Instead of: await db.users.findById(parent.authorId)
      // Use DataLoader - it batches automatically!
      return await context.loaders.user.load(parent.authorId);
    }
  }
};
```

**Setting up DataLoader:**
```javascript
const DataLoader = require('dataloader');

// Create loaders factory
function createLoaders() {
  return {
    user: new DataLoader(async (userIds) => {
      const users = await db.users.findByIds(userIds);
      const userMap = new Map(users.map(u => [u.id, u]));
      return userIds.map(id => userMap.get(id) || null);
    }),
    
    post: new DataLoader(async (postIds) => {
      const posts = await db.posts.findByIds(postIds);
      const postMap = new Map(posts.map(p => [p.id, p]));
      return postIds.map(id => postMap.get(id) || null);
    }),
    
    postsByAuthor: new DataLoader(async (authorIds) => {
      // Batch load posts for multiple authors
      const posts = await db.posts.findByAuthorIds(authorIds);
      // Group by author ID
      const postsByAuthor = new Map();
      authorIds.forEach(id => postsByAuthor.set(id, []));
      posts.forEach(post => {
        postsByAuthor.get(post.authorId).push(post);
      });
      return authorIds.map(id => postsByAuthor.get(id) || []);
    })
  };
}

// Add to context
const server = new ApolloServer({
  typeDefs,
  resolvers,
  context: ({ req }) => {
    return {
      user: getUserFromRequest(req),
      loaders: createLoaders()  // New loaders per request
    };
  }
});
```

**Using in resolvers:**
```javascript
const resolvers = {
  Query: {
    posts: async (parent, args, context) => {
      return await context.db.posts.findAll();
    }
  },
  
  Post: {
    author: async (parent, args, context) => {
      // DataLoader batches this automatically!
      return await context.loaders.user.load(parent.authorId);
    },
    
    comments: async (parent, args, context) => {
      // Can still use regular queries for non-batched data
      return await context.db.comments.findByPostId(parent.id);
    }
  },
  
  User: {
    posts: async (parent, args, context) => {
      // Batch load posts for this author
      return await context.loaders.postsByAuthor.load(parent.id);
    }
  }
};
```

**Before DataLoader (N+1 problem):**
```javascript
// Query: Get 10 posts with authors
// Makes 11 queries:
// 1. SELECT * FROM posts LIMIT 10
// 2. SELECT * FROM users WHERE id = 1
// 3. SELECT * FROM users WHERE id = 2
// ... (10 more queries)
```

**After DataLoader (batched):**
```javascript
// Query: Get 10 posts with authors
// Makes only 2 queries:
// 1. SELECT * FROM posts LIMIT 10
// 2. SELECT * FROM users WHERE id IN (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
```

**Caching:**
```javascript
const userLoader = new DataLoader(
  async (userIds) => {
    // Batch function
    const users = await db.users.findByIds(userIds);
    const userMap = new Map(users.map(u => [u.id, u]));
    return userIds.map(id => userMap.get(id) || null);
  },
  {
    cache: true,  // Default: true - caches within request
    cacheKeyFn: (key) => key.toString(),  // Custom cache key
    maxBatchSize: 100  // Maximum batch size
  }
);

// Same user loaded twice in one request? Only one query!
const user1 = await userLoader.load('123');
const user2 = await userLoader.load('123');  // Cached!
```

**Advanced: Custom batching logic:**
```javascript
const postLoader = new DataLoader(
  async (postIds) => {
    // Fetch posts
    const posts = await db.posts.findByIds(postIds);
    const postMap = new Map(posts.map(p => [p.id, p]));
    
    // Also fetch related data in the same batch
    const authorIds = [...new Set(posts.map(p => p.authorId))];
    const authors = await db.users.findByIds(authorIds);
    const authorMap = new Map(authors.map(a => [a.id, a]));
    
    // Attach authors to posts
    return postIds.map(id => {
      const post = postMap.get(id);
      if (post) {
        post.author = authorMap.get(post.authorId);
      }
      return post || null;
    });
  },
  {
    maxBatchSize: 50  // Process in batches of 50
  }
);
```

**Best practices:**
- ✅ Create new DataLoaders per request (in context)
- ✅ Use DataLoader for fields that cause N+1 problems
- ✅ Keep batch functions simple and fast
- ✅ Return results in the same order as keys
- ✅ Handle null/undefined values gracefully
- ❌ Don't use DataLoader for single, unique queries
- ❌ Don't mutate cached objects

**Performance impact:**
- **Without DataLoader:** 100 posts = 101 database queries
- **With DataLoader:** 100 posts = 2 database queries
- **Speed improvement:** 50x faster! 🚀

**Real-world scenario:**  
Like a shopping assistant who collects all your items before going to the store - instead of making 10 separate trips (N+1 queries), DataLoader batches everything into one efficient trip (batched query).

---

## Schema Design

### 22. Schema Definition Language (SDL)

**What it is:**  
SDL is the syntax for **defining GraphQL schemas** - it describes the complete API structure.

**Why learn it:**
- Understand how APIs are structured
- Design better APIs
- Document your API

**Complete Example Schema:**
```graphql
# Object Types
type User {
  id: ID!
  name: String!
  email: String
  posts: [Post!]!
  createdAt: DateTime!
}

type Post {
  id: ID!
  title: String!
  content: String!
  author: User!
  comments: [Comment!]!
  status: PostStatus!
  createdAt: DateTime!
}

type Comment {
  id: ID!
  text: String!
  author: User!
  post: Post!
  createdAt: DateTime!
}

# Root Types (entry points)
type Query {
  user(id: ID!): User
  users(limit: Int, offset: Int): [User!]!
  posts(search: String): [Post!]!
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  createPost(input: CreatePostInput!): Post!
}

type Subscription {
  postCreated: Post!
  commentAdded(postId: ID!): Comment!
}

# Input Types (for mutations)
input CreateUserInput {
  name: String!
  email: String!
  password: String!
}

input UpdateUserInput {
  name: String
  email: String
}

input CreatePostInput {
  title: String!
  content: String!
  authorId: ID!
}

# Enums
enum PostStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
}

# Interfaces
interface SearchResult {
  id: ID!
  title: String!
}

type Book implements SearchResult {
  id: ID!
  title: String!
  author: String!
  isbn: String!
}

type Article implements SearchResult {
  id: ID!
  title: String!
  author: User!
  publishedAt: DateTime!
}
```

**Key Points:**
- `type` - Defines object types
- `input` - Defines input types (for mutations)
- `enum` - Defines enumerated types
- `interface` - Defines contracts
- `Query` - Root type for queries
- `Mutation` - Root type for mutations
- `Subscription` - Root type for subscriptions
- `!` - Non-null (required)
- `[]` - List/array

---

## Best Practices

### 1. ✅ Always Use Named Operations
```graphql
# ❌ Bad
query {
  user(id: "123") { name }
}

# ✅ Good
query GetUser {
  user(id: "123") { name }
}
```
**Why:** Better debugging, logging, and error messages.

---

### 2. ✅ Use Variables Instead of String Interpolation
```graphql
# ❌ Bad - Security risk!
query {
  user(id: "${userId}") { name }
}

# ✅ Good - Type-safe and secure
query GetUser($userId: ID!) {
  user(id: $userId) { name }
}
```
**Why:** Prevents injection attacks and provides type safety.

---

### 3. ✅ Use Fragments for Reusable Fields
```graphql
# ❌ Bad - Duplication
query {
  user(id: "123") { id name email }
  users { id name email }
}

# ✅ Good - DRY principle
fragment UserDetails on User {
  id name email
}
query {
  user(id: "123") { ...UserDetails }
  users { ...UserDetails }
}
```
**Why:** Easier maintenance and consistency.

---

### 4. ✅ Request Only Needed Fields
```graphql
# ❌ Bad - Over-fetching
query {
  user(id: "123") {
    id name email bio avatar posts comments followers following
    # ... 50 more fields you don't need
  }
}

# ✅ Good - Only what you need
query {
  user(id: "123") {
    id
    name
  }
}
```
**Why:** Smaller payloads, faster responses, better performance.

---

### 5. ✅ Handle Errors Gracefully
```javascript
// Always check for errors
const { data, errors } = await client.query({ query: GET_USER });

if (errors) {
  errors.forEach(error => {
    console.error(`Error at ${error.path}: ${error.message}`);
    // Handle each error appropriately
  });
}

// Use partial data if available
if (data) {
  // Process data even if some fields failed
}
```
**Why:** Better user experience and debugging.

---

### 6. ✅ Use Type System Features
```graphql
# ✅ Use enums for fixed values
enum OrderStatus {
  PENDING
  PROCESSING
  SHIPPED
  DELIVERED
}

# ✅ Use interfaces for shared contracts
interface SearchResult {
  id: ID!
  title: String!
}

# ✅ Use non-null for required fields
type User {
  id: ID!           # Required
  name: String!     # Required
  email: String     # Optional
}
```
**Why:** Type safety, better validation, self-documenting.

---

### 7. ✅ Document Your Schema
```graphql
"""
A user account in the system.
"""
type User {
  """
  Unique identifier for the user.
  """
  id: ID!
  
  """
  User's full name.
  """
  name: String!
  
  """
  User's email address. May be null if not provided.
  """
  email: String
}

"""
Create a new user account.
"""
type Mutation {
  createUser(input: CreateUserInput!): User!
}
```
**Why:** Self-documenting API, better developer experience.

---

### 8. ✅ Use Pagination for Lists
```graphql
# ✅ Good - Paginated
query {
  users(limit: 10, offset: 0) {
    id
    name
  }
}

# Consider cursor-based pagination for large datasets
query {
  users(first: 10, after: "cursor") {
    edges {
      node {
        id
        name
      }
      cursor
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```
**Why:** Better performance and user experience.

---

## Quick Reference

### Operation Types
| Type | Purpose | Example |
|------|---------|---------|
| `query` | Read data | `query { user(id: "123") { name } }` |
| `mutation` | Modify data | `mutation { createUser(input: {...}) { id } }` |
| `subscription` | Real-time updates | `subscription { postCreated { title } }` |

### Type Modifiers
| Syntax | Meaning | Example |
|--------|---------|---------|
| `String` | Nullable | Can be `null` |
| `String!` | Non-null | Always present |
| `[String]` | Nullable list | Can be `null` or array |
| `[String!]` | List of non-nulls | Array, items can't be null |
| `[String!]!` | Non-null list of non-nulls | Always array, items can't be null |

### Common Directives
| Directive | Purpose | Example |
|-----------|---------|---------|
| `@include(if: Boolean)` | Include if true | `email @include(if: $includeEmail)` |
| `@skip(if: Boolean)` | Skip if true | `address @skip(if: $skipAddress)` |
| `@deprecated(reason: String)` | Mark as deprecated | `phone @deprecated(reason: "Use contactInfo")` |

### Built-in Scalars
| Type | Description | Example |
|------|-------------|---------|
| `Int` | 32-bit integer | `42` |
| `Float` | Double-precision float | `3.14` |
| `String` | UTF-8 string | `"Hello"` |
| `Boolean` | true/false | `true` |
| `ID` | Unique identifier | `"123"` |

### Common Patterns

**Pagination:**
```graphql
query {
  users(limit: 10, offset: 0) {
    id
    name
  }
}
```

**Filtering:**
```graphql
query {
  posts(filter: { published: true, tags: ["graphql"] }) {
    id
    title
  }
}
```

**Sorting:**
```graphql
query {
  users(sort: { field: "createdAt", order: DESC }) {
    id
    name
  }
}
```

**Nested Data:**
```graphql
query {
  user(id: "123") {
    posts {
      comments {
        author { name }
      }
    }
  }
}
```

---

## Common Use Cases

### 1. API Aggregation
Combine data from multiple sources in a single query:
```graphql
query {
  user(id: "123") {
    name
    orders { id }
    recommendations { id }
    socialMedia { followers }
  }
}
```

### 2. Mobile Optimization
Request only needed fields to reduce payload size:
```graphql
# Mobile - minimal data
query { user(id: "123") { id name } }

# Desktop - full data
query { user(id: "123") { id name email bio avatar posts } }
```

### 3. Real-time Updates
Use subscriptions for live data:
```graphql
subscription {
  orderStatusUpdated(orderId: "456") {
    status
    estimatedDelivery
  }
}
```

### 4. Type Safety
Leverage strong typing for better developer experience:
- Autocomplete in IDEs
- Catch errors before runtime
- Self-documenting APIs

### 5. API Evolution
Add new fields without breaking existing clients:
```graphql
# Old clients still work
type User {
  id: ID!
  name: String!
  # New field - optional, doesn't break old clients
  email: String
}
```

---

## Summary

GraphQL is a powerful query language that gives you:

✅ **Precise data fetching** - Get exactly what you need  
✅ **Single endpoint** - One URL for all operations  
✅ **Strong typing** - Catch errors early  
✅ **Real-time updates** - Subscriptions for live data  
✅ **Self-documenting** - Schema describes the API  
✅ **Version-free** - Evolve without breaking changes  

**Key Takeaways:**
1. Use **queries** to read data
2. Use **mutations** to modify data
3. Use **subscriptions** for real-time updates
4. Use **variables** for dynamic queries
5. Use **fragments** to avoid duplication
6. Request **only needed fields** for better performance
7. Handle **errors gracefully** for better UX
8. Leverage **type system** for safety and documentation

Happy querying! 🚀
