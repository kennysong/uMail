class Participant < ActiveRecord::Base
  has_many :experiments
end
