# This file is auto-generated from the current state of the database. Instead of editing this file, 
# please use the migrations feature of ActiveRecord to incrementally modify your database, and
# then regenerate this schema definition.
#
# Note that this schema.rb definition is the authoritative source for your database schema. If you need
# to create the application database on another system, you should be using db:schema:load, not running
# all the migrations from scratch. The latter is a flawed and unsustainable approach (the more migrations
# you'll amass, the slower it'll run and the greater likelihood for issues).
#
# It's strongly recommended to check this file into your version control system.

ActiveRecord::Schema.define(:version => 8) do

  create_table "email_threads", :force => true do |t|
    t.string  "Name"
    t.boolean "Lock",   :default => false
    t.string  "listno"
  end

  create_table "emails", :force => true do |t|
    t.string   "To"
    t.string   "From"
    t.datetime "Date"
    t.string   "Subject"
    t.string   "Cc"
    t.text     "Body"
    t.boolean  "Hidden",          :default => false
    t.integer  "email_thread_id"
    t.string   "Submit_By"
    t.datetime "Submit_Date"
    t.text     "Original"
  end

  create_table "experiments", :force => true do |t|
    t.string   "Admin"
    t.integer  "participant_id"
    t.string   "Location"
    t.datetime "Date"
  end

  create_table "participants", :force => true do |t|
    t.string  "Name"
    t.string  "Email"
    t.string  "Class"
    t.string  "Department"
    t.boolean "Again"
    t.string  "Recruitment"
  end

  create_table "summaries", :force => true do |t|
    t.integer "email_thread_id"
    t.integer "experiment_id"
    t.text    "Sum"
    t.string  "Sent"
    t.string  "Label"
    t.integer "list_order",      :default => 1
  end

  create_table "users", :force => true do |t|
    t.string   "login"
    t.string   "email"
    t.string   "crypted_password",          :limit => 40
    t.string   "salt",                      :limit => 40
    t.datetime "created_at"
    t.datetime "updated_at"
    t.string   "remember_token"
    t.datetime "remember_token_expires_at"
  end

end
