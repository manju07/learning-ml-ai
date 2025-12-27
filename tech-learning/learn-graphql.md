# GraphQL Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Intermediate Concepts](#intermediate-concepts)
4. [Advanced Concepts](#advanced-concepts)
5. [Schema Design](#schema-design)
6. [Best Practices](#best-practices)
7. [Quick Reference](#quick-reference)

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
    tags            # [String!]! - always an array (may be empty)
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
- `tags` is **always** an array (even if empty `[]`)
- `posts` **might** be null OR an array
- `friends` is **always** an array of user objects

**Real-world scenario:**  
Like TypeScript types - you know exactly what to expect, and the compiler catches errors before runtime.

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
Unions allow a field to return **one of several possible types**, even if they don't share a common interface.

**Why use it:**  
- Flexible return types
- Handle unrelated types
- Type-safe queries

**Example:**
```graphql
query {
  content(id: "123") {
    ... on BlogPost {
      title
      content
      author
      publishedAt
    }
    ... on Video {
      title
      url
      duration
      thumbnail
    }
    ... on Podcast {
      title
      audioUrl
      episode
      host
    }
  }
}
```

**What happens:**  
The `content` field can return a `BlogPost`, `Video`, or `Podcast`. You use inline fragments to handle each type.

**Interfaces vs Unions:**
- **Interfaces:** Types share common fields (like a contract)
- **Unions:** Types might be completely different (just grouped together)

**Real-world scenario:**  
Like a media player that can play different file types (MP3, MP4, WAV) - they're different formats, but you handle them all in one place.

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
