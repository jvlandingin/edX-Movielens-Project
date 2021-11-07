library(tidyverse)
library(stringr)
library(caret)
library(data.table)

######################
#Creating the datasets
######################

#Creating the edx and validation set
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#saving data files
save(edx, file = "edx.rda")
save(validation, file = "validation.rda")

#Creating the train_set and test_set to be used for model development
test_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = F)
train_set <- edx[-test_index,]
temp <- edx[test_index,]
test_set <- temp %>%
  semi_join(train_set, by = 'movieId') %>%
  semi_join(train_set, by = 'userId')

save(train_set, file = "train_set.rda")
save(test_set, file = "test_set.rda")


#The following is the function that computes for the RMSE:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

############################
#Creating the effects/biases
############################

mu <- mean(train_set$rating)

#Movie effect
movie_effect <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating-mu)/(n() + 1.75))


##User effect
user_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  group_by(userId) %>% 
  mutate(pred = mu + b_i) %>%
  summarize(b_u = sum(rating-pred)/(n() + 5))


##Genre effect
###Getting least squares estimate was done similar with user effect and movie effect
###Penalty terms (lambda) are also already added which gave minimum RMSEs when cross-validated
drama_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  mutate(drama = ifelse(str_detect(genres, "Drama"), 1, 0)) %>%
  filter(drama == 1) %>%
  group_by(userId) %>%
  summarize(b_drama = sum(rating - pred)/(n() + 27))

comedy_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0)
  ) %>%
  mutate(comedy = ifelse(str_detect(genres, "Comedy"), 1, 0)) %>%
  filter(comedy == 1) %>%
  group_by(userId) %>%
  summarize(b_comedy = sum(rating - pred)/(n() + 32))

action_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0)
  ) %>%
  mutate(action = ifelse(str_detect(genres, "Action"), 1, 0)) %>%
  filter(action == 1) %>%
  group_by(userId) %>%
  summarize(b_action = sum(rating - pred)/(n() + 11))

thriller_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0)
  ) %>%
  mutate(thriller = ifelse(str_detect(genres, "Thriller"), 1, 0)) %>%
  filter(thriller == 1) %>%
  group_by(userId) %>%
  summarize(b_thriller = sum(rating - pred)/(n() + 36))

adventure_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0)
  ) %>%
  mutate(adventure = ifelse(str_detect(genres, "Adventure"), 1, 0)) %>%
  filter(adventure == 1) %>%
  group_by(userId) %>%
  summarize(b_adventure = sum(rating - pred)/(n() + 27))

romance_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0)
  ) %>%
  mutate(romance = ifelse(str_detect(genres, "Romance"), 1, 0)) %>%
  filter(romance == 1) %>%
  group_by(userId) %>%
  summarize(b_romance = sum(rating - pred)/(n() + 31))

sci_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0)
  ) %>%
  mutate(sci = ifelse(str_detect(genres, "Sci"), 1, 0)) %>%
  filter(sci == 1) %>%
  group_by(userId) %>%
  summarize(b_sci = sum(rating - pred)/(n() + 25))

crime_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0)
  ) %>%
  mutate(crime = ifelse(str_detect(genres, "Crime"), 1, 0)) %>%
  filter(crime == 1) %>%
  group_by(userId) %>%
  summarize(b_crime = sum(rating - pred)/(n() + 25))

fantasy_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0)
  ) %>%
  mutate(fantasy = ifelse(str_detect(genres, "Fantasy"), 1, 0)) %>%
  filter(fantasy == 1) %>%
  group_by(userId) %>%
  summarize(b_fantasy = sum(rating - pred)/(n() + 25))

children_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  left_join(fantasy_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0)
  ) %>%
  mutate(children = ifelse(str_detect(genres, "Children"), 1, 0)) %>%
  filter(children == 1) %>%
  group_by(userId) %>%
  summarize(b_children = sum(rating - pred)/(n() + 7))

horror_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  left_join(fantasy_effect, by='userId') %>%
  left_join(children_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0) +
           ifelse(str_detect(genres, "Children") & !is.na(b_children), b_children, 0)
  ) %>%
  mutate(horror = ifelse(str_detect(genres, "Horror"), 1, 0)) %>%
  filter(horror == 1) %>%
  group_by(userId) %>%
  summarize(b_horror = sum(rating - pred)/(n() + 9))

mystery_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  left_join(fantasy_effect, by='userId') %>%
  left_join(children_effect, by='userId') %>%
  left_join(horror_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0) +
           ifelse(str_detect(genres, "Children") & !is.na(b_children), b_children, 0) +
           ifelse(str_detect(genres, "Horror") & !is.na(b_horror), b_horror, 0)
  ) %>%
  mutate(mystery = ifelse(str_detect(genres, "Mystery"), 1, 0)) %>%
  filter(mystery == 1) %>%
  group_by(userId) %>%
  summarize(b_mystery = sum(rating - pred)/(n() + 62))

war_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  left_join(fantasy_effect, by='userId') %>%
  left_join(children_effect, by='userId') %>%
  left_join(horror_effect, by='userId') %>%
  left_join(mystery_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0) +
           ifelse(str_detect(genres, "Children") & !is.na(b_children), b_children, 0) +
           ifelse(str_detect(genres, "Horror") & !is.na(b_horror), b_horror, 0) +
           ifelse(str_detect(genres, "Mystery") & !is.na(b_mystery), b_mystery, 0)
  ) %>%
  mutate(war = ifelse(str_detect(genres, "War"), 1, 0)) %>%
  filter(war == 1) %>%
  group_by(userId) %>%
  summarize(b_war = sum(rating - pred)/(n() + 28))

animation_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  left_join(fantasy_effect, by='userId') %>%
  left_join(children_effect, by='userId') %>%
  left_join(horror_effect, by='userId') %>%
  left_join(mystery_effect, by='userId') %>%
  left_join(war_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0) +
           ifelse(str_detect(genres, "Children") & !is.na(b_children), b_children, 0) +
           ifelse(str_detect(genres, "Horror") & !is.na(b_horror), b_horror, 0) +
           ifelse(str_detect(genres, "Mystery") & !is.na(b_mystery), b_mystery, 0) +
           ifelse(str_detect(genres, "War") & !is.na(b_war), b_war, 0)
  ) %>%
  mutate(animation = ifelse(str_detect(genres, "Animation"), 1, 0)) %>%
  filter(animation == 1) %>%
  group_by(userId) %>%
  summarize(b_animation = sum(rating - pred)/(n() + 11))

musical_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  left_join(fantasy_effect, by='userId') %>%
  left_join(children_effect, by='userId') %>%
  left_join(horror_effect, by='userId') %>%
  left_join(mystery_effect, by='userId') %>%
  left_join(war_effect, by='userId') %>%
  left_join(animation_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0) +
           ifelse(str_detect(genres, "Children") & !is.na(b_children), b_children, 0) +
           ifelse(str_detect(genres, "Horror") & !is.na(b_horror), b_horror, 0) +
           ifelse(str_detect(genres, "Mystery") & !is.na(b_mystery), b_mystery, 0) +
           ifelse(str_detect(genres, "War") & !is.na(b_war), b_war, 0) +
           ifelse(str_detect(genres, "Animation") & !is.na(b_animation), b_animation, 0)
  ) %>%
  mutate(musical = ifelse(str_detect(genres, "Musical"), 1, 0)) %>%
  filter(musical == 1) %>%
  group_by(userId) %>%
  summarize(b_musical = sum(rating - pred)/(n() + 16))

western_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  left_join(fantasy_effect, by='userId') %>%
  left_join(children_effect, by='userId') %>%
  left_join(horror_effect, by='userId') %>%
  left_join(mystery_effect, by='userId') %>%
  left_join(war_effect, by='userId') %>%
  left_join(animation_effect, by='userId') %>%
  left_join(musical_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0) +
           ifelse(str_detect(genres, "Children") & !is.na(b_children), b_children, 0) +
           ifelse(str_detect(genres, "Horror") & !is.na(b_horror), b_horror, 0) +
           ifelse(str_detect(genres, "Mystery") & !is.na(b_mystery), b_mystery, 0) +
           ifelse(str_detect(genres, "War") & !is.na(b_war), b_war, 0) +
           ifelse(str_detect(genres, "Animation") & !is.na(b_animation), b_animation, 0) +
           ifelse(str_detect(genres, "Musical") & !is.na(b_musical), b_musical, 0)
  ) %>%
  mutate(western = ifelse(str_detect(genres, "Western"), 1, 0)) %>%
  filter(western == 1) %>%
  group_by(userId) %>%
  summarize(b_western = sum(rating - pred)/(n() + 20))

film_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  left_join(fantasy_effect, by='userId') %>%
  left_join(children_effect, by='userId') %>%
  left_join(horror_effect, by='userId') %>%
  left_join(mystery_effect, by='userId') %>%
  left_join(war_effect, by='userId') %>%
  left_join(animation_effect, by='userId') %>%
  left_join(musical_effect, by='userId') %>%
  left_join(western_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0) +
           ifelse(str_detect(genres, "Children") & !is.na(b_children), b_children, 0) +
           ifelse(str_detect(genres, "Horror") & !is.na(b_horror), b_horror, 0) +
           ifelse(str_detect(genres, "Mystery") & !is.na(b_mystery), b_mystery, 0) +
           ifelse(str_detect(genres, "War") & !is.na(b_war), b_war, 0) +
           ifelse(str_detect(genres, "Animation") & !is.na(b_animation), b_animation, 0) +
           ifelse(str_detect(genres, "Musical") & !is.na(b_musical), b_musical, 0) +
           ifelse(str_detect(genres, "Western") & !is.na(b_western), b_western, 0)
  ) %>%
  mutate(film = ifelse(str_detect(genres, "Film"), 1, 0)) %>%
  filter(film == 1) %>%
  group_by(userId) %>%
  summarize(b_film = sum(rating - pred)/(n() + 9))

documentary_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  left_join(fantasy_effect, by='userId') %>%
  left_join(children_effect, by='userId') %>%
  left_join(horror_effect, by='userId') %>%
  left_join(mystery_effect, by='userId') %>%
  left_join(war_effect, by='userId') %>%
  left_join(animation_effect, by='userId') %>%
  left_join(musical_effect, by='userId') %>%
  left_join(western_effect, by='userId') %>%
  left_join(film_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0) +
           ifelse(str_detect(genres, "Children") & !is.na(b_children), b_children, 0) +
           ifelse(str_detect(genres, "Horror") & !is.na(b_horror), b_horror, 0) +
           ifelse(str_detect(genres, "Mystery") & !is.na(b_mystery), b_mystery, 0) +
           ifelse(str_detect(genres, "War") & !is.na(b_war), b_war, 0) +
           ifelse(str_detect(genres, "Animation") & !is.na(b_animation), b_animation, 0) +
           ifelse(str_detect(genres, "Musical") & !is.na(b_musical), b_musical, 0) +
           ifelse(str_detect(genres, "Western") & !is.na(b_western), b_western, 0) +
           ifelse(str_detect(genres, "Film") & !is.na(b_film), b_film, 0)
  ) %>%
  mutate(documentary = ifelse(str_detect(genres, "Documentary"), 1, 0)) %>%
  filter(documentary == 1) %>%
  group_by(userId) %>%
  summarize(b_documentary = sum(rating - pred)/(n() + 4))

imax_effect <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  left_join(fantasy_effect, by='userId') %>%
  left_join(children_effect, by='userId') %>%
  left_join(horror_effect, by='userId') %>%
  left_join(mystery_effect, by='userId') %>%
  left_join(war_effect, by='userId') %>%
  left_join(animation_effect, by='userId') %>%
  left_join(musical_effect, by='userId') %>%
  left_join(western_effect, by='userId') %>%
  left_join(film_effect, by='userId') %>%
  left_join(documentary_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0) +
           ifelse(str_detect(genres, "Children") & !is.na(b_children), b_children, 0) +
           ifelse(str_detect(genres, "Horror") & !is.na(b_horror), b_horror, 0) +
           ifelse(str_detect(genres, "Mystery") & !is.na(b_mystery), b_mystery, 0) +
           ifelse(str_detect(genres, "War") & !is.na(b_war), b_war, 0) +
           ifelse(str_detect(genres, "Animation") & !is.na(b_animation), b_animation, 0) +
           ifelse(str_detect(genres, "Musical") & !is.na(b_musical), b_musical, 0) +
           ifelse(str_detect(genres, "Western") & !is.na(b_western), b_western, 0) +
           ifelse(str_detect(genres, "Film") & !is.na(b_film), b_film, 0) +
           ifelse(str_detect(genres, "Documentary") & !is.na(b_documentary), b_documentary, 0)
  ) %>%
  mutate(imax = ifelse(str_detect(genres, "IMAX"), 1, 0)) %>%
  filter(imax == 1) %>%
  group_by(userId) %>%
  summarize(b_imax = sum(rating - pred)/(n() + 5.5))

#Below is the table with the Movie+User+Genre Effects
mmug_table <- train_set %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by='userId') %>%
  left_join(drama_effect, by='userId') %>%
  left_join(comedy_effect, by='userId') %>%
  left_join(action_effect, by='userId') %>%
  left_join(thriller_effect, by='userId') %>%
  left_join(adventure_effect, by='userId') %>%
  left_join(romance_effect, by='userId') %>%
  left_join(sci_effect, by='userId') %>%
  left_join(crime_effect, by='userId') %>%
  left_join(fantasy_effect, by='userId') %>%
  left_join(children_effect, by='userId') %>%
  left_join(horror_effect, by='userId') %>%
  left_join(mystery_effect, by='userId') %>%
  left_join(war_effect, by='userId') %>%
  left_join(animation_effect, by='userId') %>%
  left_join(musical_effect, by='userId') %>%
  left_join(western_effect, by='userId') %>%
  left_join(film_effect, by='userId') %>%
  left_join(documentary_effect, by='userId') %>%
  left_join(imax_effect, by='userId') %>%
  mutate(pred = mu + b_i + b_u +
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) +
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0) +
           ifelse(str_detect(genres, "Children") & !is.na(b_children), b_children, 0) +
           ifelse(str_detect(genres, "Horror") & !is.na(b_horror), b_horror, 0) +
           ifelse(str_detect(genres, "Mystery") & !is.na(b_mystery), b_mystery, 0) +
           ifelse(str_detect(genres, "War") & !is.na(b_war), b_war, 0) +
           ifelse(str_detect(genres, "Animation") & !is.na(b_animation), b_animation, 0) +
           ifelse(str_detect(genres, "Musical") & !is.na(b_musical), b_musical, 0) +
           ifelse(str_detect(genres, "Western") & !is.na(b_western), b_western, 0) +
           ifelse(str_detect(genres, "Film") & !is.na(b_film), b_film, 0) +
           ifelse(str_detect(genres, "Documentary") & !is.na(b_documentary), b_documentary, 0) +
           ifelse(str_detect(genres, "IMAX") & !is.na(b_imax), b_imax, 0)
  )

##Movie release year effect
movieYear <- mmug_table %>% 
  mutate(timestamp = year(as.Date.POSIXct(timestamp)),
         movieYear = as.numeric(str_match(title, "(\\d{4})\\)$")[,2])) %>%
  group_by(movieYear) %>%
  summarize(b_movieYear = mean(rating - pred), n = n())

###K-nearest neighbor was used here to predict rating from movie release year
train_knn <- train(b_movieYear ~ ., method = "knn", 
                   tuneGrid = data.frame(k = seq(1,51,2)), 
                   data = movieYear)
predict_knn <- predict(train_knn, movieYear)

movieYear_effect <- movieYear %>% 
  mutate(knn = predict_knn)

####################
#Final Hold-Out Test
####################

#Then we do the final hold-out test using the code below
pred <- validation %>% 
  left_join(movie_effect, by = 'movieId') %>%
  left_join(user_effect, by = 'userId') %>%
  left_join(comedy_effect, by = 'userId') %>%
  left_join(drama_effect, by = 'userId') %>%
  left_join(action_effect, by = 'userId') %>%
  left_join(thriller_effect, by = 'userId') %>%
  left_join(adventure_effect, by = 'userId') %>%
  left_join(romance_effect, by = 'userId') %>%
  left_join(sci_effect, by = 'userId') %>%
  left_join(crime_effect, by = 'userId') %>%
  left_join(fantasy_effect, by = 'userId') %>%
  left_join(children_effect, by = 'userId') %>%
  left_join(horror_effect, by = 'userId') %>%
  left_join(mystery_effect, by = 'userId') %>%
  left_join(war_effect, by = 'userId') %>%
  left_join(animation_effect, by = 'userId') %>%
  left_join(musical_effect, by = 'userId') %>%
  left_join(western_effect, by = 'userId') %>%
  left_join(film_effect, by = 'userId') %>%
  left_join(documentary_effect, by = 'userId') %>%
  left_join(imax_effect, by = 'userId') %>%
  mutate(b_u = ifelse(is.na(b_u), 0, b_u),
         b_i = ifelse(is.na(b_i), 0, b_i),
         pred = mu + b_i + b_u,
         newpred = pred + 
           ifelse(str_detect(genres, "Comedy") & !is.na(b_comedy), b_comedy, 0) + 
           ifelse(str_detect(genres, "Drama") & !is.na(b_drama), b_drama, 0) +
           ifelse(str_detect(genres, "Action") & !is.na(b_action), b_action, 0) +
           ifelse(str_detect(genres, "Thriller") & !is.na(b_thriller), b_thriller, 0) +
           ifelse(str_detect(genres, "Adventure") & !is.na(b_adventure), b_adventure, 0) +
           ifelse(str_detect(genres, "Romance") & !is.na(b_romance), b_romance, 0) +
           ifelse(str_detect(genres, "Sci") & !is.na(b_sci), b_sci, 0) + 
           ifelse(str_detect(genres, "Crime") & !is.na(b_crime), b_crime, 0) +
           ifelse(str_detect(genres, "Fantasy") & !is.na(b_fantasy), b_fantasy, 0) +
           ifelse(str_detect(genres, "Children") & !is.na(b_children), b_children, 0) +
           ifelse(str_detect(genres, "Horror") & !is.na(b_horror), b_horror, 0) +
           ifelse(str_detect(genres, "Mystery") & !is.na(b_mystery), b_mystery, 0) +
           ifelse(str_detect(genres, "War") & !is.na(b_war), b_war, 0) +
           ifelse(str_detect(genres, "Animation") & !is.na(b_animation), b_animation, 0) +
           ifelse(str_detect(genres, "Musical") & !is.na(b_musical), b_musical, 0) +
           ifelse(str_detect(genres, "Western") & !is.na(b_western), b_western, 0) +
           ifelse(str_detect(genres, "Film") & !is.na(b_film), b_film, 0) +
           ifelse(str_detect(genres, "Documentary") & !is.na(b_documentary), b_documentary, 0) +
           ifelse(str_detect(genres, "IMAX") & !is.na(b_imax), b_imax, 0)
  ) %>%
  mutate(movieYear = as.numeric(str_match(title, "(\\d{4})\\)$")[,2])) %>%
  left_join(movieYear_effect) %>% 
  mutate(newpred = newpred + knn) %>%
  .$newpred

RMSE(pred, validation$rating)